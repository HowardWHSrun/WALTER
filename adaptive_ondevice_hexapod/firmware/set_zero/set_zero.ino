/*
  set_zero — Pololu Maestro: drive all channels to neutral ("zero") pulse width.

  Teensy 4.1 → Serial1 @ 9600 → Maestro (18-channel: channels 0–17).

  "Zero" here means each channel’s calibrated center in microseconds (same idea
  as servoCenter[] in hexapod_walking_v2). Default is 1500 µs everywhere; edit
  servoCenter[] to match your hardware. First 12 entries mirror the walking sketch.
*/

#define maestroSerial Serial1

const int NUM_SERVOS = 18;

// Per-channel center pulse (µs). Channels 12–17 are extras for Maestro 18; tune as needed.
int servoCenter[NUM_SERVOS] = {
  1500, 1500,   // 0–1  FL abd, flx
  1500, 1500,   // 2–3  FR abd, flx
  1500, 1500,   // 4–5  ML abd, flx
  1500, 1500,   // 6–7  MR abd, flx
  1500, 1500,   // 8–9  RL abd, flx
  1500, 1500,   // 10–11 RR abd, flx
  1500, 1500, 1500, 1500, 1500, 1500  // 12–17 spare / future
};

const int MIN_US = 600;
const int MAX_US = 2400;

int clampUs(int us) {
  if (us < MIN_US) return MIN_US;
  if (us > MAX_US) return MAX_US;
  return us;
}

void setMaestroTarget(uint8_t channel, uint16_t targetUs) {
  uint16_t target = targetUs * 4;
  maestroSerial.write(0x84);
  maestroSerial.write(channel);
  maestroSerial.write(target & 0x7F);
  maestroSerial.write((target >> 7) & 0x7F);
}

void writeAllCenters() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    setMaestroTarget(i, (uint16_t)clampUs(servoCenter[i]));
  }
}

void setup() {
  maestroSerial.begin(9600);
  delay(1500);

  maestroSerial.write(0xAA);
  delay(100);

  for (int i = 0; i < 3; i++) {
    maestroSerial.write(0xA1);
    delay(50);
  }

  writeAllCenters();
}

void loop() {
  // Hold position; Maestro keeps last targets. Add delay/repeat here if you need a periodic refresh.
}
