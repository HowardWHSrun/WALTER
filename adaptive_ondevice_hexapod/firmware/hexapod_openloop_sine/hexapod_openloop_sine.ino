/*
 * Open-loop sinusoidal tripod gait (no IMU) — same phase/rest/shift structure as
 * hexapod_brain_roadmap copy/ode_optimization_no_imu/hexapod_no_imu_optimization.py
 *
 * Optimizer outputs angles in degrees; jointToUs() applies those as *microsecond
 * offsets* from servoCenter. ABD_CMD_PER_DEG / FLX_CMD_PER_DEG must stay small enough
 * that (deg × scale) stays within ~±400–500 µs typical travel — not ~42 µs/deg on flex.
 *
 * Leg order: FL, FR, ML, MR, RL, RR (Maestro channels 0..11, abd then flx per leg).
 * Tripod A: FL, MR, RL (phase 0); Tripod B: FR, ML, RR (phase pi).
 */

#include <math.h>

#include "optimized_gait_params.h"

#define maestroSerial Serial1

#ifndef PI
#define PI 3.14159265358979323846
#endif

enum ServoID {
  FL_ABD = 0, FL_FLX = 1,
  FR_ABD = 2, FR_FLX = 3,
  ML_ABD = 4, ML_FLX = 5,
  MR_ABD = 6, MR_FLX = 7,
  RL_ABD = 8, RL_FLX = 9,
  RR_ABD = 10, RR_FLX = 11
};

const int NUM_SERVOS = 12;

int servoCenter[NUM_SERVOS] = {
  1500, 1500,
  1500, 1500,
  1500, 1500,
  1500, 1500,
  1500, 1500,
  1500, 1500
};

// v2 mirror: left (+1,-1), right (-1,+1). Using (-1,+1) on *every* channel removes
// L/R mirroring and often looks "twisted" even if a single-leg test looked backward.
// Fix left issues with the multipliers below (±1 only). Try FLX first, then ABD, then both.
static const int L_ABD = +1;
static const int L_FLX = -1;
static const int R_ABD = -1;
static const int R_FLX = +1;
static const int LEFT_FLIP_ABD = +1;  // -1 to invert abduction on FL, ML, RL only
static const int LEFT_FLIP_FLX = +1;  // -1 to invert flex on FL, ML, RL only

int servoDir[NUM_SERVOS] = {
  L_ABD * LEFT_FLIP_ABD, L_FLX * LEFT_FLIP_FLX,
  R_ABD, R_FLX,
  L_ABD * LEFT_FLIP_ABD, L_FLX * LEFT_FLIP_FLX,
  R_ABD, R_FLX,
  L_ABD * LEFT_FLIP_ABD, L_FLX * LEFT_FLIP_FLX,
  R_ABD, R_FLX,
};

// 1: Serial @115200 — pulse samples + clip counts every ~500 ms (disable for silent runs).
#ifndef OPENLOOP_DIAG_SERIAL
#define OPENLOOP_DIAG_SERIAL 1
#endif
#ifndef OPENLOOP_CLIP_REPORT
#define OPENLOOP_CLIP_REPORT OPENLOOP_DIAG_SERIAL
#endif
// Pause (µs) after each Maestro Set Target; helps some serial links.
#ifndef MAESTRO_CMD_GAP_US
#define MAESTRO_CMD_GAP_US 120
#endif

int currentUs[NUM_SERVOS];

const int MIN_US = 600;
const int MAX_US = 2400;

// ms between gait updates (same order of magnitude as v2 INTERP_DELAY)
const unsigned INTERP_DELAY_MS = 10;

// Degrees → µs offset at servoCenter (keep modest so jointCommand stays ~hundreds of µs).
const float ABD_CMD_PER_DEG = 8.0f;
const float FLX_CMD_PER_DEG = 6.0f;

// Limits on abstract *joint commands* (µs offset before servoDir). Tight FLX max avoids
// huge excursions when scale × degrees was miscalibrated.
const int ABD_CMD_MIN = -400;
const int ABD_CMD_MAX = 400;
const int FLX_CMD_MIN = -400;
const int FLX_CMD_MAX = 480;

// Ramp 0 → target frequency over this many ms at boot (smoother bring-up).
const unsigned long FREQ_RAMP_MS = 4000;
// If > 0, caps gait Hz after ramp: min(OPT_FREQUENCY_HZ, cap). Use 1.0f for calmer bench;0 for full OPT.
static const float BENCH_FREQ_CAP_HZ = 1.0f;

// Tripod phase offset per leg index: FL, FR, ML, MR, RL, RR
const float TRIPOD_OFFSET_RAD[6] = {
  0.f,
  (float)PI,
  (float)PI,
  0.f,
  0.f,
  (float)PI
};

unsigned long lastTickMs = 0;
double phase = 0.0;
unsigned long bootMs = 0;

#if OPENLOOP_CLIP_REPORT
static uint16_t g_clipAbdCmd;
static uint16_t g_clipFlxCmd;
static uint16_t g_clipUs;
#endif

void setMaestroTarget(uint8_t channel, uint16_t targetUs) {
  uint16_t target = targetUs * 4;
  maestroSerial.write(0x84);
  maestroSerial.write(channel);
  maestroSerial.write(target & 0x7F);
  maestroSerial.write((target >> 7) & 0x7F);
}

int clampUs(int us) {
  if (us < MIN_US) return MIN_US;
  if (us > MAX_US) return MAX_US;
  return us;
}

int jointToUs(uint8_t servo, int jointCommand) {
  return clampUs(servoCenter[servo] + servoDir[servo] * jointCommand);
}

/** Like jointToUs; if clipCount non-null, increments when output hits MIN_US/MAX_US. */
int jointToUsClip(uint8_t servo, int jointCommand, uint16_t* clipCount) {
  int raw = servoCenter[servo] + servoDir[servo] * jointCommand;
  int us = clampUs(raw);
  if (clipCount != nullptr && us != raw) {
    (*clipCount)++;
  }
  return us;
}

int clampAbdCmd(int c) {
  if (c < ABD_CMD_MIN) return ABD_CMD_MIN;
  if (c > ABD_CMD_MAX) return ABD_CMD_MAX;
  return c;
}

int clampFlxCmd(int c) {
  if (c < FLX_CMD_MIN) return FLX_CMD_MIN;
  if (c > FLX_CMD_MAX) return FLX_CMD_MAX;
  return c;
}

void writeAll() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    setMaestroTarget(i, currentUs[i]);
#if MAESTRO_CMD_GAP_US > 0
    delayMicroseconds(MAESTRO_CMD_GAP_US);
#endif
  }
}

void buildPose(
  int targetUs[NUM_SERVOS],
  int fl_abd, int fl_flx,
  int fr_abd, int fr_flx,
  int ml_abd, int ml_flx,
  int mr_abd, int mr_flx,
  int rl_abd, int rl_flx,
  int rr_abd, int rr_flx,
  uint16_t* usClipCount
) {
  targetUs[FL_ABD] = jointToUsClip(FL_ABD, fl_abd, usClipCount);
  targetUs[FL_FLX] = jointToUsClip(FL_FLX, fl_flx, usClipCount);
  targetUs[FR_ABD] = jointToUsClip(FR_ABD, fr_abd, usClipCount);
  targetUs[FR_FLX] = jointToUsClip(FR_FLX, fr_flx, usClipCount);
  targetUs[ML_ABD] = jointToUsClip(ML_ABD, ml_abd, usClipCount);
  targetUs[ML_FLX] = jointToUsClip(ML_FLX, ml_flx, usClipCount);
  targetUs[MR_ABD] = jointToUsClip(MR_ABD, mr_abd, usClipCount);
  targetUs[MR_FLX] = jointToUsClip(MR_FLX, mr_flx, usClipCount);
  targetUs[RL_ABD] = jointToUsClip(RL_ABD, rl_abd, usClipCount);
  targetUs[RL_FLX] = jointToUsClip(RL_FLX, rl_flx, usClipCount);
  targetUs[RR_ABD] = jointToUsClip(RR_ABD, rr_abd, usClipCount);
  targetUs[RR_FLX] = jointToUsClip(RR_FLX, rr_flx, usClipCount);
}

static float gaitFrequencyTargetHz() {
  float f = OPT_FREQUENCY_HZ;
  if (BENCH_FREQ_CAP_HZ > 0.f && f > BENCH_FREQ_CAP_HZ) {
    f = BENCH_FREQ_CAP_HZ;
  }
  return f;
}

/** Effective frequency during optional startup ramp */
float frequencyNow(unsigned long nowMs) {
  float target = gaitFrequencyTargetHz();
  if (FREQ_RAMP_MS == 0) {
    return target;
  }
  unsigned long t = nowMs - bootMs;
  if (t >= FREQ_RAMP_MS) {
    return target;
  }
  return target * (float)t / (float)FREQ_RAMP_MS;
}

void applySineGaitTick(float dt) {
  float f = frequencyNow(millis());
  phase += 2.0 * PI * (double)f * (double)dt;
  while (phase >= 2.0 * PI) phase -= 2.0 * PI;
  while (phase < 0.0) phase += 2.0 * PI;

  int abdCmd[6];
  int flxCmd[6];

  for (int i = 0; i < 6; i++) {
    float legBase = (float)phase + TRIPOD_OFFSET_RAD[i];
    float abdPh = legBase + OPT_ABD_SHIFT_DEG[i] * (float)(PI / 180.0);
    float flxPh = legBase + OPT_FLX_SHIFT_DEG[i] * (float)(PI / 180.0);

    float abdDeg = OPT_ABD_REST_DEG[i] + OPT_ABD_AMP_DEG[i] * sinf(abdPh);
    float flxDeg = OPT_FLX_REST_DEG[i] + OPT_FLX_AMP_DEG[i] * sinf(flxPh);

    // Swing lift (matches Python LIFT_SWING_DEG_PEAK): pull foot up when sin(abd)>0 — tune sign on hardware if needed
    float sinAbd = sinf(abdPh);
    float liftBlend = (sinAbd > 0.f) ? sinAbd : 0.f;
    flxDeg -= OPT_LIFT_SWING_DEG_PEAK * liftBlend;

    int jAbd = (int)lroundf(abdDeg * ABD_CMD_PER_DEG);
    int jFlx = (int)lroundf(flxDeg * FLX_CMD_PER_DEG);
    abdCmd[i] = clampAbdCmd(jAbd);
    flxCmd[i] = clampFlxCmd(jFlx);
#if OPENLOOP_CLIP_REPORT
    if (abdCmd[i] != jAbd) {
      g_clipAbdCmd++;
    }
    if (flxCmd[i] != jFlx) {
      g_clipFlxCmd++;
    }
#endif
  }

  uint16_t usClips = 0;
  int target[NUM_SERVOS];
  buildPose(
    target,
    abdCmd[0], flxCmd[0],
    abdCmd[1], flxCmd[1],
    abdCmd[2], flxCmd[2],
    abdCmd[3], flxCmd[3],
    abdCmd[4], flxCmd[4],
    abdCmd[5], flxCmd[5],
    OPENLOOP_CLIP_REPORT ? &usClips : nullptr
  );

#if OPENLOOP_CLIP_REPORT
  g_clipUs += usClips;
#endif

  for (int s = 0; s < NUM_SERVOS; s++) {
    currentUs[s] = target[s];
  }
#if OPENLOOP_DIAG_SERIAL
  static unsigned long _diagMs = 0;
  unsigned long _now = millis();
  if (_now - _diagMs >= 500) {
    _diagMs = _now;
    Serial.print(F("us L: FL "));
    Serial.print(target[FL_ABD]);
    Serial.print(F(","));
    Serial.print(target[FL_FLX]);
    Serial.print(F(" ML "));
    Serial.print(target[ML_ABD]);
    Serial.print(F(","));
    Serial.print(target[ML_FLX]);
    Serial.print(F(" RL "));
    Serial.print(target[RL_ABD]);
    Serial.print(F(","));
    Serial.print(target[RL_FLX]);
    Serial.print(F(" | R: FR "));
    Serial.print(target[FR_ABD]);
    Serial.print(F(","));
    Serial.print(target[FR_FLX]);
#if OPENLOOP_CLIP_REPORT
    Serial.print(F(" | clips(cmd abd,flx / us)="));
    Serial.print(g_clipAbdCmd);
    Serial.print(F(","));
    Serial.print(g_clipFlxCmd);
    Serial.print(F("/"));
    Serial.print(g_clipUs);
    g_clipAbdCmd = 0;
    g_clipFlxCmd = 0;
    g_clipUs = 0;
#endif
    Serial.println();
  }
#endif
  writeAll();
}

void setup() {
#if OPENLOOP_DIAG_SERIAL
  Serial.begin(115200);
  delay(500);
#endif
  bootMs = millis();
  lastTickMs = bootMs;

  maestroSerial.begin(9600);
  delay(1500);

  maestroSerial.write(0xAA);
  delay(100);

  for (int i = 0; i < 3; i++) {
    maestroSerial.write(0xA1);
    delay(50);
  }

  for (int i = 0; i < NUM_SERVOS; i++) {
    currentUs[i] = clampUs(servoCenter[i]);
  }
  writeAll();
  delay(500);

  // Brief hold at rest pose (sin t=0) before continuous motion
  applySineGaitTick(0.f);
  delay(300);
}

void loop() {
  unsigned long now = millis();
  unsigned long dtMs = now - lastTickMs;
  if (dtMs < INTERP_DELAY_MS) return;

  lastTickMs = now;
  float dt = (float)dtMs / 1000.0f;
  if (dt > 0.05f) dt = 0.05f;

  applySineGaitTick(dt);
}
