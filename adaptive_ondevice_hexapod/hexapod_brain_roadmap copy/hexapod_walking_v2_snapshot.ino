/*
================================================================================
12-DOF HEXAPOD TRIPOD GAIT CONTROLLER  —  v2
================================================================================

Changes from v1:
  1. Body rides higher at rest  (START_FLEX raised from 1100 → 1400)
  2. Swing phase now truly LIFTS the leg off the ground (new LIFT_FLEX = -300)
  3. Gait restructured into   lift → swing → plant   sub-phases
     so the foot clears the ground, travels forward in the air,
     then plants before the stance legs push.
  4. Push phase slightly reduced (1850 → 1650) to avoid bottoming out
     at the higher ride height.

Everything else (channel map, Maestro protocol, mirroring, baseline
offsets) is unchanged from v1.

================================================================================
*/

#define maestroSerial Serial1

// ============================================================================
// ENUM: SERVO IDs
// ============================================================================

enum ServoID {
  FL_ABD = 0, FL_FLX = 1,
  FR_ABD = 2, FR_FLX = 3,
  ML_ABD = 4, ML_FLX = 5,
  MR_ABD = 6, MR_FLX = 7,
  RL_ABD = 8, RL_FLX = 9,
  RR_ABD = 10, RR_FLX = 11
};

const int NUM_SERVOS = 12;

// ============================================================================
// SERVO CENTER CALIBRATION  (microseconds)
// ============================================================================

int servoCenter[NUM_SERVOS] = {
  1500, 1500,   // FL_ABD, FL_FLX
  1500, 1500,   // FR_ABD, FR_FLX
  1500, 1500,   // ML_ABD, ML_FLX
  1500, 1500,   // MR_ABD, MR_FLX
  1500, 1500,   // RL_ABD, RL_FLX
  1500, 1500    // RR_ABD, RR_FLX
};

// ============================================================================
// SERVO DIRECTION CALIBRATION
// ============================================================================

int servoDir[NUM_SERVOS] = {
  +1, -1,   // FL_ABD, FL_FLX
  -1, +1,   // FR_ABD, FR_FLX
  +1, -1,   // ML_ABD, ML_FLX
  -1, +1,   // MR_ABD, MR_FLX
  +1, -1,   // RL_ABD, RL_FLX
  -1, +1    // RR_ABD, RR_FLX
};

// ============================================================================
// CURRENT SERVO COMMANDS
// ============================================================================

int currentUs[NUM_SERVOS];

// ============================================================================
// BASELINE ABDUCTION OFFSETS  (standing geometry — unchanged from v1)
// ============================================================================

const int FRONT_ABD_OFFSET = 180;
const int MID_ABD_OFFSET   = 0;
const int REAR_ABD_OFFSET  = -180;

// ============================================================================
// DYNAMIC ABDUCTION GAIT AMPLITUDES
// ----------------------------------------------------------------------------
// Slightly increased to take advantage of the fact that feet now truly
// clear the ground during swing.
// ============================================================================

const int FRONT_SWING_ABD = 110;    // was 90  — longer stride now that feet lift
const int FRONT_PUSH_ABD  = -110;   // was -90

const int MID_SWING_ABD   = 85;     // was 70
const int MID_PUSH_ABD    = -85;    // was -70

const int REAR_SWING_ABD  = 110;    // was 90
const int REAR_PUSH_ABD   = -110;   // was -90

// ============================================================================
// FLEXION / EXTENSION GAIT TUNING
// ----------------------------------------------------------------------------
// Positive FLX = more DOWN.
//
// v2 changes:
//   START_FLEX   1100 → 1400   Higher resting body height.
//   PUSH_FLEX    1850 → 1650   Strong push without bottoming out.
//   LIFT_FLEX    (new) -120    Slightly reduced lift versus original -300.
//   PLANT_FLEX   (new) 1400    Re-plants to normal stance height.
//
// The old RELEASE_FLEX (700) only "lightened" the leg — it did not
// lift the foot, so the robot dragged.  LIFT_FLEX fixes that.
// ============================================================================

const int START_FLEX = 1400;    // neutral standing — body is higher now
const int PUSH_FLEX  = 1650;    // stance legs push hard (near original behavior)
const int LIFT_FLEX  = -120;    // swing legs lift less, but still clear ground
const int PLANT_FLEX = 1400;    // same as START — foot re-plants at ride height

// ============================================================================
// GAIT SPEED / SMOOTHING
// ----------------------------------------------------------------------------
// Several step counts are used for different sub-phases so timing
// can be tuned independently.
// ============================================================================

const int INTERP_DELAY = 10;    // ms per interpolation step  (unchanged)

const int LIFT_STEPS   = 10;    // fast lift so foot clears quickly
const int SWING_STEPS  = 16;    // moderate speed forward swing
const int PLANT_STEPS  = 8;     // quick replant
const int PUSH_STEPS   = 16;    // matches swing so stride is symmetric
const int MID_STEPS    = 12;    // transition back to neutral

// ============================================================================
// SERVO SAFETY CLAMP
// ============================================================================

const int MIN_US = 600;
const int MAX_US = 2400;

// ============================================================================
// LOW-LEVEL MAESTRO WRITE
// ============================================================================

void setMaestroTarget(uint8_t channel, uint16_t targetUs) {
  uint16_t target = targetUs * 4;

  maestroSerial.write(0x84);
  maestroSerial.write(channel);
  maestroSerial.write(target & 0x7F);
  maestroSerial.write((target >> 7) & 0x7F);
}

// ============================================================================
// CLAMP TO SAFE RANGE
// ============================================================================

int clampUs(int us) {
  if (us < MIN_US) return MIN_US;
  if (us > MAX_US) return MAX_US;
  return us;
}

// ============================================================================
// ABSTRACT JOINT COMMAND → REAL SERVO PULSE
// ============================================================================

int jointToUs(uint8_t servo, int jointCommand) {
  return clampUs(servoCenter[servo] + servoDir[servo] * jointCommand);
}

// ============================================================================
// WRITE CURRENT POSE TO ALL SERVOS
// ============================================================================

void writeAll() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    setMaestroTarget(i, currentUs[i]);
  }
}

// ============================================================================
// SMOOTH INTERPOLATED MOVE
// ============================================================================

void moveSmoothTo(int targetUs[NUM_SERVOS], int steps, int delayMs) {
  int startUs[NUM_SERVOS];

  for (int i = 0; i < NUM_SERVOS; i++) {
    startUs[i] = currentUs[i];
  }

  for (int s = 1; s <= steps; s++) {
    for (int i = 0; i < NUM_SERVOS; i++) {
      currentUs[i] = startUs[i] + (targetUs[i] - startUs[i]) * s / steps;
    }
    writeAll();
    delay(delayMs);
  }
}

// ============================================================================
// POSE BUILDER
// ============================================================================

void buildPose(
  int targetUs[NUM_SERVOS],
  int fl_abd, int fl_flx,
  int fr_abd, int fr_flx,
  int ml_abd, int ml_flx,
  int mr_abd, int mr_flx,
  int rl_abd, int rl_flx,
  int rr_abd, int rr_flx
) {
  targetUs[FL_ABD] = jointToUs(FL_ABD, fl_abd);
  targetUs[FL_FLX] = jointToUs(FL_FLX, fl_flx);

  targetUs[FR_ABD] = jointToUs(FR_ABD, fr_abd);
  targetUs[FR_FLX] = jointToUs(FR_FLX, fr_flx);

  targetUs[ML_ABD] = jointToUs(ML_ABD, ml_abd);
  targetUs[ML_FLX] = jointToUs(ML_FLX, ml_flx);

  targetUs[MR_ABD] = jointToUs(MR_ABD, mr_abd);
  targetUs[MR_FLX] = jointToUs(MR_FLX, mr_flx);

  targetUs[RL_ABD] = jointToUs(RL_ABD, rl_abd);
  targetUs[RL_FLX] = jointToUs(RL_FLX, rl_flx);

  targetUs[RR_ABD] = jointToUs(RR_ABD, rr_abd);
  targetUs[RR_FLX] = jointToUs(RR_FLX, rr_flx);
}

// ============================================================================
// BASELINE ABDUCTION HELPERS
// ============================================================================

int flBaseAbd() { return FRONT_ABD_OFFSET; }
int frBaseAbd() { return FRONT_ABD_OFFSET; }
int mlBaseAbd() { return MID_ABD_OFFSET;   }
int mrBaseAbd() { return MID_ABD_OFFSET;   }
int rlBaseAbd() { return REAR_ABD_OFFSET;  }
int rrBaseAbd() { return REAR_ABD_OFFSET;  }

// ============================================================================
// DOWN STANCE  (initial standing posture — now taller)
// ============================================================================

void downStance() {
  int target[NUM_SERVOS];

  buildPose(
    target,
    flBaseAbd(), START_FLEX,
    frBaseAbd(), START_FLEX,
    mlBaseAbd(), START_FLEX,
    mrBaseAbd(), START_FLEX,
    rlBaseAbd(), START_FLEX,
    rrBaseAbd(), START_FLEX
  );

  moveSmoothTo(target, 30, INTERP_DELAY);   // slow rise on boot
}

// ============================================================================
// TRIPOD A — LIFT
// ----------------------------------------------------------------------------
// Tripod A legs (FL, MR, RL) lift off the ground.
// Tripod B legs (FR, ML, RR) stay planted at stance height.
// ABD stays at baseline — no forward/backward motion yet.
// ============================================================================

void tripodA_lift() {
  int target[NUM_SERVOS];

  buildPose(
    target,
    flBaseAbd(), LIFT_FLEX,       // FL lifts
    frBaseAbd(), START_FLEX,      // FR stays planted

    mlBaseAbd(), START_FLEX,      // ML stays planted
    mrBaseAbd(), LIFT_FLEX,       // MR lifts

    rlBaseAbd(), LIFT_FLEX,       // RL lifts
    rrBaseAbd(), START_FLEX       // RR stays planted
  );

  moveSmoothTo(target, LIFT_STEPS, INTERP_DELAY);
}

// ============================================================================
// TRIPOD A — SWING FORWARD  +  TRIPOD B — PUSH BACKWARD
// ----------------------------------------------------------------------------
// While A legs are still in the air, swing them forward.
// Simultaneously, B legs push backward against the ground — this is
// what actually propels the body forward.
// ============================================================================

void tripodA_swingB_push() {
  int target[NUM_SERVOS];

  buildPose(
    target,
    flBaseAbd() + FRONT_SWING_ABD, LIFT_FLEX,      // FL swings fwd, still lifted
    frBaseAbd() + FRONT_PUSH_ABD,  PUSH_FLEX,      // FR pushes back, loaded

    mlBaseAbd() + MID_PUSH_ABD,    PUSH_FLEX,      // ML pushes back, loaded
    mrBaseAbd() + MID_SWING_ABD,   LIFT_FLEX,      // MR swings fwd, still lifted

    rlBaseAbd() + REAR_SWING_ABD,  LIFT_FLEX,      // RL swings fwd, still lifted
    rrBaseAbd() + REAR_PUSH_ABD,   PUSH_FLEX       // RR pushes back, loaded
  );

  moveSmoothTo(target, SWING_STEPS, INTERP_DELAY);
}

// ============================================================================
// TRIPOD A — PLANT
// ----------------------------------------------------------------------------
// A legs have arrived at their forward position.
// Now push them back down to the ground so they can become stance legs.
// B legs hold position.
// ============================================================================

void tripodA_plant() {
  int target[NUM_SERVOS];

  buildPose(
    target,
    flBaseAbd() + FRONT_SWING_ABD, PLANT_FLEX,     // FL plants down at fwd pos
    frBaseAbd() + FRONT_PUSH_ABD,  PUSH_FLEX,      // FR still pushing

    mlBaseAbd() + MID_PUSH_ABD,    PUSH_FLEX,      // ML still pushing
    mrBaseAbd() + MID_SWING_ABD,   PLANT_FLEX,     // MR plants down at fwd pos

    rlBaseAbd() + REAR_SWING_ABD,  PLANT_FLEX,     // RL plants down at fwd pos
    rrBaseAbd() + REAR_PUSH_ABD,   PUSH_FLEX       // RR still pushing
  );

  moveSmoothTo(target, PLANT_STEPS, INTERP_DELAY);
}

// ============================================================================
// TRIPOD B — LIFT
// ============================================================================

void tripodB_lift() {
  int target[NUM_SERVOS];

  buildPose(
    target,
    flBaseAbd() + FRONT_SWING_ABD, START_FLEX,     // FL stays planted (fwd pos)
    frBaseAbd() + FRONT_PUSH_ABD,  LIFT_FLEX,      // FR lifts

    mlBaseAbd() + MID_PUSH_ABD,    LIFT_FLEX,      // ML lifts
    mrBaseAbd() + MID_SWING_ABD,   START_FLEX,     // MR stays planted (fwd pos)

    rlBaseAbd() + REAR_SWING_ABD,  START_FLEX,     // RL stays planted (fwd pos)
    rrBaseAbd() + REAR_PUSH_ABD,   LIFT_FLEX       // RR lifts
  );

  moveSmoothTo(target, LIFT_STEPS, INTERP_DELAY);
}

// ============================================================================
// TRIPOD B — SWING FORWARD  +  TRIPOD A — PUSH BACKWARD
// ============================================================================

void tripodB_swingA_push() {
  int target[NUM_SERVOS];

  buildPose(
    target,
    flBaseAbd() + FRONT_PUSH_ABD,  PUSH_FLEX,      // FL pushes back, loaded
    frBaseAbd() + FRONT_SWING_ABD, LIFT_FLEX,      // FR swings fwd, still lifted

    mlBaseAbd() + MID_SWING_ABD,   LIFT_FLEX,      // ML swings fwd, still lifted
    mrBaseAbd() + MID_PUSH_ABD,    PUSH_FLEX,      // MR pushes back, loaded

    rlBaseAbd() + REAR_PUSH_ABD,   PUSH_FLEX,      // RL pushes back, loaded
    rrBaseAbd() + REAR_SWING_ABD,  LIFT_FLEX       // RR swings fwd, still lifted
  );

  moveSmoothTo(target, SWING_STEPS, INTERP_DELAY);
}

// ============================================================================
// TRIPOD B — PLANT
// ============================================================================

void tripodB_plant() {
  int target[NUM_SERVOS];

  buildPose(
    target,
    flBaseAbd() + FRONT_PUSH_ABD,  PUSH_FLEX,      // FL still pushing
    frBaseAbd() + FRONT_SWING_ABD, PLANT_FLEX,     // FR plants down at fwd pos

    mlBaseAbd() + MID_SWING_ABD,   PLANT_FLEX,     // ML plants down at fwd pos
    mrBaseAbd() + MID_PUSH_ABD,    PUSH_FLEX,      // MR still pushing

    rlBaseAbd() + REAR_PUSH_ABD,   PUSH_FLEX,      // RL still pushing
    rrBaseAbd() + REAR_SWING_ABD,  PLANT_FLEX      // RR plants down at fwd pos
  );

  moveSmoothTo(target, PLANT_STEPS, INTERP_DELAY);
}

// ============================================================================
// NEUTRAL MID-STANCE  (optional — called between full cycles)
// ----------------------------------------------------------------------------
// Brings all legs back to baseline.  You can remove this from the loop
// for continuous walking, or keep it for a brief pause between strides.
// ============================================================================

void neutralStance() {
  int target[NUM_SERVOS];

  buildPose(
    target,
    flBaseAbd(), START_FLEX,
    frBaseAbd(), START_FLEX,
    mlBaseAbd(), START_FLEX,
    mrBaseAbd(), START_FLEX,
    rlBaseAbd(), START_FLEX,
    rrBaseAbd(), START_FLEX
  );

  moveSmoothTo(target, MID_STEPS, INTERP_DELAY);
}

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  maestroSerial.begin(9600);
  delay(1500);

  maestroSerial.write(0xAA);   // auto-baud sync
  delay(100);

  for (int i = 0; i < 3; i++) {
    maestroSerial.write(0xA1);   // clear errors / exit safe start
    delay(50);
  }

  for (int i = 0; i < NUM_SERVOS; i++) {
    currentUs[i] = clampUs(servoCenter[i]);
  }

  writeAll();
  delay(1000);

  downStance();
  delay(500);
}

// ============================================================================
// MAIN LOOP
// ----------------------------------------------------------------------------
// Full gait cycle:
//
//   1. Tripod A lifts          — feet clear the ground
//   2. Tripod A swings fwd     — feet travel forward through the air
//      Tripod B pushes back    — body is propelled forward
//   3. Tripod A plants         — feet touch down at the new forward position
//
//   4. Tripod B lifts
//   5. Tripod B swings fwd
//      Tripod A pushes back
//   6. Tripod B plants
//
// This is a continuous walk.  Remove neutralStance() for non-stop motion,
// or keep it for a brief settle between strides.
// ============================================================================

void loop() {
  // --- stride 1: Tripod A in air, Tripod B on ground ---
  tripodA_lift();
  tripodA_swingB_push();
  tripodA_plant();

  // --- stride 2: Tripod B in air, Tripod A on ground ---
  tripodB_lift();
  tripodB_swingA_push();
  tripodB_plant();
}
