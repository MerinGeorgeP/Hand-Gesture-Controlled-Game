extends CharacterBody2D

# ============================================================
# PLAYER SETTINGS
# ============================================================

const SPEED = 400.0
const JUMP_VELOCITY = -900.0

@onready var sprite_2d = $Sprite2D


# ============================================================
# UDP SETTINGS
# ============================================================

var udp := PacketPeerUDP.new()

const UDP_PORT = 4242

# Current gesture received from Python
var active_gesture := "NONE"

# Time since last gesture packet
var gesture_timer := 0.0

# If Python stops sending gestures, stop the player
const GESTURE_TIMEOUT = 0.25


# ============================================================
# READY
# ============================================================

func _ready() -> void:

	var err = udp.bind(UDP_PORT)

	if err == OK:
		print("==========================================")
		print("UDP HAND GESTURE CONTROLLER")
		print("Listening on port: ", UDP_PORT)
		print("==========================================")
	else:
		print("ERROR: Failed to bind UDP port ", UDP_PORT)
		print("Error code: ", err)


# ============================================================
# PHYSICS
# ============================================================

func _physics_process(delta: float) -> void:

	# --------------------------------------------------------
	# READ UDP PACKETS FROM PYTHON
	# --------------------------------------------------------

	while udp.get_available_packet_count() > 0:

		var packet = udp.get_packet()

		var message = packet.get_string_from_utf8().strip_edges().to_upper()

		if message != "":
			active_gesture = message
			gesture_timer = 0.0

			print("Gesture received: ", active_gesture)


	# --------------------------------------------------------
	# GESTURE TIMEOUT
	# --------------------------------------------------------

	gesture_timer += delta

	if gesture_timer > GESTURE_TIMEOUT:
		active_gesture = "NONE"


	# --------------------------------------------------------
	# KEYBOARD MOVEMENT
	# --------------------------------------------------------

	var direction := Input.get_axis("ui_left", "ui_right")

	var gesture_jump := false


	# --------------------------------------------------------
	# HAND GESTURE CONTROLS
	#
	# I = FORWARD
	# L = BACKWARD
	# X = JUMP
	# Q = FORWARD + JUMP
	# W = BACKWARD + JUMP
	# --------------------------------------------------------

	match active_gesture:

		"I":
			direction = 1.0

		"L":
			direction = -1.0

		"X":
			gesture_jump = true

		"Q":
			direction = 1.0
			gesture_jump = true

		"W":
			direction = -1.0
			gesture_jump = true

		"NONE":
			# No hand command
			pass


	# --------------------------------------------------------
	# GRAVITY
	# --------------------------------------------------------

	if not is_on_floor():

		velocity += get_gravity() * delta

		if sprite_2d:
			sprite_2d.animation = "jumping"


	# --------------------------------------------------------
	# JUMP
	# --------------------------------------------------------

	if (Input.is_action_just_pressed("ui_accept") or gesture_jump) and is_on_floor():

		velocity.y = JUMP_VELOCITY


	# --------------------------------------------------------
	# RUNNING / IDLE ANIMATION
	# --------------------------------------------------------

	if is_on_floor():

		if abs(velocity.x) > 1:

			if sprite_2d:
				sprite_2d.animation = "running"

		else:

			if sprite_2d:
				sprite_2d.animation = "default"


	# --------------------------------------------------------
	# HORIZONTAL MOVEMENT
	# --------------------------------------------------------

	if direction != 0:

		velocity.x = direction * SPEED

	else:

		velocity.x = move_toward(
			velocity.x,
			0,
			SPEED
		)


	# --------------------------------------------------------
	# FLIP SPRITE
	# --------------------------------------------------------

	if velocity.x != 0:

		if sprite_2d:
			sprite_2d.flip_h = velocity.x < 0


	# --------------------------------------------------------
	# MOVE PLAYER
	# --------------------------------------------------------

	move_and_slide()
