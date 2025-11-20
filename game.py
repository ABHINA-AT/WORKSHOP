import cv2
import numpy as np
import random
import time
from math import hypot

# --- 1. CONFIGURATION (tweak these) ---
LOWER_BLUE = np.array([100, 50, 50])
UPPER_BLUE = np.array([130, 255, 255])
TRACKING_COLOR_BGR = (255, 0, 0)  # trail color

SCORE = 0
MAX_TRAIL_POINTS = 20
TARGET_RADIUS = 30
NUM_TARGETS = 6

# Toss / physics
GRAVITY = 0.6
MIN_INIT_VY = -25
MAX_INIT_VY = -18
MIN_INIT_VX = -4
MAX_INIT_VX = 4

# Slicing threshold: require the tracked object to be moving faster than this (pixels/frame)
SLICE_SPEED_THRESHOLD = 12

# Palette of colors (B, G, R tuples) for targets
COLOR_PALETTE = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
]

# Bomb settings
BOMB_RADIUS = 28
BOMB_SPAWN_INTERVAL = 6.0   # seconds between bomb spawns
MAX_BOMBS_ON_SCREEN = 2

# Visual effect durations (seconds)
HIT_EFFECT_DURATION = 0.06
BOMB_EXPLOSION_EFFECT_DURATION = 0.12

# Morphology kernel for mask cleanup
KERNEL = np.ones((5, 5), np.uint8)

# --- 2. INITIALIZATION ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

point_trail = []
targets = []
bombs = []
targets_initialized = False

last_bomb_time = 0.0
game_over = False
game_over_time = None

# visual effect lists store tuples (x, y, expire_time, kind)
# kind: 'hit' or 'explosion'
visual_effects = []

def respawn_target(t, W, H):
    """Spawn a new target from below the screen with upward initial velocity."""
    t["x"] = float(random.randint(50, max(50, W - 50)))
    t["y"] = float(H + TARGET_RADIUS + random.randint(20, 120))  # start slightly off-screen (below)
    t["vx"] = float(random.uniform(MIN_INIT_VX, MAX_INIT_VX))
    t["vy"] = float(random.uniform(MIN_INIT_VY, MAX_INIT_VY))
    t["alive"] = True
    t["spawn_time"] = time.time()
    t["color"] = random.choice(COLOR_PALETTE)

def spawn_bomb(b, W, H):
    """Spawn a bomb similar to a target but visually different."""
    b["x"] = float(random.randint(80, max(80, W - 80)))
    b["y"] = float(H + BOMB_RADIUS + random.randint(40, 140))
    b["vx"] = float(random.uniform(MIN_INIT_VX, MAX_INIT_VX))
    b["vy"] = float(random.uniform(MIN_INIT_VY - 4, MAX_INIT_VY - 2))
    b["alive"] = True
    b["spawn_time"] = time.time()
    b["color"] = (50, 50, 50)

def reset_game(W, H):
    """Reset game state and initialize targets."""
    global SCORE, point_trail, targets, bombs, last_bomb_time, game_over, game_over_time, visual_effects
    SCORE = 0
    point_trail = []
    targets = []
    bombs = []
    last_bomb_time = time.time()
    game_over = False
    game_over_time = None
    visual_effects = []
    for _ in range(NUM_TARGETS):
        t = {}
        respawn_target(t, W, H)
        # give some variety to starting y
        t["y"] += float(random.randint(0, 150))
        targets.append(t)

# --- 3. GAME LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    if not targets_initialized:
        reset_game(W, H)
        targets_initialized = True

    now = time.time()

    if not game_over:
        # spawn bombs at intervals (cap number on screen)
        if now - last_bomb_time >= BOMB_SPAWN_INTERVAL and len(bombs) < MAX_BOMBS_ON_SCREEN:
            b = {}
            spawn_bomb(b, W, H)
            bombs.append(b)
            last_bomb_time = now

        # --- detect blue object ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
        mask = cv2.erode(mask, KERNEL, iterations=2)
        mask = cv2.dilate(mask, KERNEL, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center = None
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 8:
                center = (int(x), int(y))
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                point_trail.append(center)

        # keep trail limited
        if len(point_trail) > MAX_TRAIL_POINTS:
            point_trail = point_trail[-MAX_TRAIL_POINTS:]

        # draw trail lines
        for i in range(1, len(point_trail)):
            if point_trail[i - 1] is None or point_trail[i] is None:
                continue
            thickness = int(np.sqrt(MAX_TRAIL_POINTS / float(i + 1)) * 2.0)
            cv2.line(frame, point_trail[i - 1], point_trail[i], TRACKING_COLOR_BGR, thickness)

        # compute simple slice speed (based on last two centers)
        slice_speed = 0.0
        if len(point_trail) >= 2 and point_trail[-1] is not None and point_trail[-2] is not None:
            dx = float(point_trail[-1][0] - point_trail[-2][0])
            dy = float(point_trail[-1][1] - point_trail[-2][1])
            slice_speed = hypot(dx, dy)

        # --- update targets: physics (arc) + draw + collision (slice) ---
        for t in targets:
            # update position by velocity
            t["x"] += t["vx"]
            t["y"] += t["vy"]

            # apply gravity (pull down)
            t["vy"] += GRAVITY

            # draw target using its own color (ensure ints for drawing)
            tx, ty = int(round(t["x"])), int(round(t["y"]))
            if 0 <= tx <= W and 0 <= ty <= H:
                cv2.circle(frame, (tx, ty), TARGET_RADIUS, t["color"], -1)
                cv2.circle(frame, (tx, ty), TARGET_RADIUS, (0, 0, 0), 2)
            else:
                # draw off-screen indicators lightly (optional)
                cv2.circle(frame, (max(0, min(W - 1, tx)), max(0, min(H - 1, ty))), TARGET_RADIUS, t["color"], 1)

            # If the target has fallen back below the screen (missed) or gone far sideways -> respawn it.
            if t["y"] - TARGET_RADIUS > H + 50 or t["x"] < -200 or t["x"] > W + 200:
                respawn_target(t, W, H)
                continue

            # collision with player's tracked object: only count as hit when slicing fast enough
            if center is not None:
                dist = hypot(center[0] - t["x"], center[1] - t["y"])
                if dist < TARGET_RADIUS and slice_speed >= SLICE_SPEED_THRESHOLD:
                    SCORE += 1
                    # add a brief visual hit effect (non-blocking)
                    visual_effects.append((t["x"], t["y"], now + HIT_EFFECT_DURATION, "hit"))
                    respawn_target(t, W, H)

        # --- update bombs: physics + draw + collision (slice ends game) ---
        for b in bombs[:]:  # iterate over a shallow copy because we may remove items
            b["x"] += b["vx"]
            b["y"] += b["vy"]
            b["vy"] += GRAVITY

            bx, by = int(round(b["x"])), int(round(b["y"]))
            # draw bomb: dark circle + little X for visibility
            cv2.circle(frame, (bx, by), BOMB_RADIUS, (40, 40, 40), -1)
            cv2.circle(frame, (bx, by), BOMB_RADIUS, (0, 0, 0), 2)
            cv2.line(frame, (int(b["x"] - BOMB_RADIUS/1.5), int(b["y"] - BOMB_RADIUS/1.5)),
                     (int(b["x"] + BOMB_RADIUS/1.5), int(b["y"] + BOMB_RADIUS/1.5)), (0, 0, 255), 2)
            cv2.line(frame, (int(b["x"] - BOMB_RADIUS/1.5), int(b["y"] + BOMB_RADIUS/1.5)),
                     (int(b["x"] + BOMB_RADIUS/1.5), int(b["y"] - BOMB_RADIUS/1.5)), (0, 0, 255), 2)

            # remove bombs that left the screen (so new bombs can spawn later)
            if b["y"] - BOMB_RADIUS > H + 80 or b["x"] < -200 or b["x"] > W + 200:
                bombs.remove(b)
                continue

            # collision check: if sliced -> game over
            if center is not None:
                dist_b = hypot(center[0] - b["x"], center[1] - b["y"])
                if dist_b < BOMB_RADIUS and slice_speed >= SLICE_SPEED_THRESHOLD:
                    # explosion effect (visual) and set game over (non-blocking)
                    visual_effects.append((b["x"], b["y"], now + BOMB_EXPLOSION_EFFECT_DURATION, "explosion"))
                    game_over = True
                    game_over_time = time.time()
                    print("BOOM! Game Over.")
                    break  # stop processing more bombs/targets this frame

        # show score and info while playing
        cv2.putText(frame, f"Score: {SCORE}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Slice speed: {int(slice_speed)}", (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    else:
        # GAME OVER display - freeze gameplay, but allow restart
        cv2.putText(frame, "GAME OVER", (max(10, W//2 - 180), max(10, H//2 - 20)),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 4)
        cv2.putText(frame, f"Final Score: {SCORE}", (max(10, W//2 - 160), max(10, H//2 + 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, "Press 'r' to restart or 'q' to quit", (max(10, W//2 - 260), max(10, H//2 + 100)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # --- draw visual effects (hits, explosions) ---
    now = time.time()
    new_effects = []
    for (ex, ey, expire, kind) in visual_effects:
        if now < expire:
            ex_i, ey_i = int(round(ex)), int(round(ey))
            if kind == "hit":
                cv2.circle(frame, (ex_i, ey_i), TARGET_RADIUS + 12, (255, 255, 255), 4)
            elif kind == "explosion":
                cv2.circle(frame, (ex_i, ey_i), BOMB_RADIUS + 20, (0, 0, 255), 6)
            new_effects.append((ex, ey, expire, kind))
    visual_effects = new_effects

    cv2.imshow("Fruit-Ninja Toss Targets (Bombs Enabled)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r') and game_over:
        reset_game(W, H)

cap.release()
cv2.destroyAllWindows()