/*
 * Diagnostic: right side only (FR, MR, RR). Left legs (FL, ML, RL) held at center.
 *
 * Cycles every RIGHT_LEG_CYCLE_MS:
 *   FR (Maestro 2,3) -> MR (6,7) -> RR (10,11)
 * Each active leg: ABD and FLX follow a slow sine (same phase) so both joints visibly move.
 *
 * Upload to Teensy 4.1; USB Serial 115200 prints which leg is active.
 * Maestro on Serial1 @ 9600 (same as walking / openloop sketches).
 */

#include <math.h>

#define maestroSerial Serial1

#ifndef PI
#define PI 3.14159265358979323846
#endif

enum ServoID {
  FL_ABD = 0,
  FL_FLX = 1,
  FR_ABD = 2,
  FR_FLX = 3,
  ML_ABD = 4,
  ML_FLX = 5,
  MR_ABD = 6,
  MR_FLX = 7,
  RL_ABD = 8,
  RL_FLX = 9,
  RR_ABD = 10,
  RR_FLX = 11
};

const int NUM_SERVOS = 12;

int servoCenter[NUM_SERVOS] = {
  1500, 1500, 1500, 1500, 1500, 1500,
  1500, 1500, 1500, 1500, 1500, 1500
};

// Match firmware v2 default L/R mirroring; change here if your walking sketch differs.
int servoDir[NUM_SERVOS] = {
  +1, -1,   // FL
  -1, +1,   // FR
  +1, -1,   // ML
  -1, +1,   // MR
  +1, -1,   // RL
  -1, +1    // RR
};

const int MIN_US = 600;
const int MAX_US = 2400;

// How long to exercise each right leg before switching (ms)
const unsigned long RIGHT_LEG_CYCLE_MS = 5000;
// Oscillation frequency (Hz) while a leg is active
const float OSC_HZ = 0.35f;
// Joint-command amplitude (same units as openloop sketch; keep moderate)
const int ABD_SWING = 90;
const int FLX_SWING = 90;

const unsigned INTERP_DELAY_MS = 15;

unsigned long lastTickMs = 0;
unsigned long lastPrintMs = 0;

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

/** Which right leg is active: 0 = FR, 1 = MR, 2 = RR */
int activeRightLegIndex(unsigned long t) {
  unsigned long slot = t / RIGHT_LEG_CYCLE_MS;
  return (int)(slot % 3);
}

bool channelIsRightSide(uint8_t ch) {
  return (ch >= FR_ABD && ch <= FR_FLX) || (ch >= MR_ABD && ch <= MR_FLX) ||
         (ch >= RR_ABD && ch <= RR_FLX);
}

bool channelIsActiveRightLeg(uint8_t ch, int legIdx) {
  if (legIdx == 0) return (ch == FR_ABD || ch == FR_FLX);
  if (legIdx == 1) return (ch == MR_ABD || ch == MR_FLX);
  return (ch == RR_ABD || ch == RR_FLX);
}

void writeAll(const int targetUs[NUM_SERVOS]) {
  for (int i = 0; i < NUM_SERVOS; i++) {
    setMaestroTarget((uint8_t)i, (uint16_t)targetUs[i]);
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
  }

  lastTickMs = millis();

  maestroSerial.begin(9600);
  delay(1500);

  maestroSerial.write(0xAA);
  delay(100);
  for (int i = 0; i < 3; i++) {
    maestroSerial.write(0xA1);
    delay(50);
  }

  int neutral[NUM_SERVOS];
  for (int i = 0; i < NUM_SERVOS; i++) {
    neutral[i] = clampUs(servoCenter[i]);
  }
  writeAll(neutral);
  delay(400);

  Serial.println(F("hexapod_test_right_side_only: LEFT frozen at center."));
  Serial.println(F("Right cycles FR -> MR -> RR. Watch USB for active leg."));
}

void loop() {
  unsigned long now = millis();
  if (now - lastTickMs < INTERP_DELAY_MS) {
    return;
  }
  float dt = (float)(now - lastTickMs) / 1000.0f;
  lastTickMs = now;
  if (dt > 0.08f) {
    dt = 0.08f;
  }

  static double phase = 0.0;
  phase += 2.0 * PI * (double)OSC_HZ * (double)dt;
  while (phase >= 2.0 * PI) {
    phase -= 2.0 * PI;
  }

  int legIdx = activeRightLegIndex(now);
  float s = sinf((float)phase);
  int abdCmd = (int)lroundf(s * (float)ABD_SWING);
  int flxCmd = (int)lroundf(s * (float)FLX_SWING);

  int target[NUM_SERVOS];
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (!channelIsRightSide((uint8_t)i)) {
      target[i] = clampUs(servoCenter[i]);
      continue;
    }
    if (!channelIsActiveRightLeg((uint8_t)i, legIdx)) {
      target[i] = jointToUs((uint8_t)i, 0);
      continue;
    }
    if (i % 2 == 0) {
      target[i] = jointToUs((uint8_t)i, abdCmd);
    } else {
      target[i] = jointToUs((uint8_t)i, flxCmd);
    }
  }

  writeAll(target);

  if (now - lastPrintMs >= 800) {
    lastPrintMs = now;
    const char *name = (legIdx == 0) ? "FR" : (legIdx == 1) ? "MR" : "RR";
    Serial.print(F("Active right leg: "));
    Serial.print(name);
    Serial.print(F("  (slot "));
    Serial.print(now / RIGHT_LEG_CYCLE_MS);
    Serial.println(F(")"));
  }
}
