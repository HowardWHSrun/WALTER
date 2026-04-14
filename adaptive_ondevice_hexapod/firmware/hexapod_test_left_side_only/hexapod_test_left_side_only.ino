/*
 * Diagnostic: left side only (FL, ML, RL). Right legs (FR, MR, RR) held at center.
 *
 * Cycles every LEFT_LEG_CYCLE_MS:
 *   FL (Maestro 0,1) -> ML (4,5) -> RL (8,9)
 * Pair with hexapod_test_right_side_only to see which side misbehaves.
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

const unsigned long LEFT_LEG_CYCLE_MS = 5000;
const float OSC_HZ = 0.35f;
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

int activeLeftLegIndex(unsigned long t) {
  unsigned long slot = t / LEFT_LEG_CYCLE_MS;
  return (int)(slot % 3);
}

bool channelIsLeftSide(uint8_t ch) {
  return (ch >= FL_ABD && ch <= FL_FLX) || (ch >= ML_ABD && ch <= ML_FLX) ||
         (ch >= RL_ABD && ch <= RL_FLX);
}

bool channelIsActiveLeftLeg(uint8_t ch, int legIdx) {
  if (legIdx == 0) return (ch == FL_ABD || ch == FL_FLX);
  if (legIdx == 1) return (ch == ML_ABD || ch == ML_FLX);
  return (ch == RL_ABD || ch == RL_FLX);
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

  Serial.println(F("hexapod_test_left_side_only: RIGHT frozen at center."));
  Serial.println(F("Left cycles FL -> ML -> RL."));
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

  int legIdx = activeLeftLegIndex(now);
  float s = sinf((float)phase);
  int abdCmd = (int)lroundf(s * (float)ABD_SWING);
  int flxCmd = (int)lroundf(s * (float)FLX_SWING);

  int target[NUM_SERVOS];
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (!channelIsLeftSide((uint8_t)i)) {
      target[i] = clampUs(servoCenter[i]);
      continue;
    }
    if (!channelIsActiveLeftLeg((uint8_t)i, legIdx)) {
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
    const char *name = (legIdx == 0) ? "FL" : (legIdx == 1) ? "ML" : "RL";
    Serial.print(F("Active left leg: "));
    Serial.print(name);
    Serial.print(F("  (slot "));
    Serial.print(now / LEFT_LEG_CYCLE_MS);
    Serial.println(F(")"));
  }
}
