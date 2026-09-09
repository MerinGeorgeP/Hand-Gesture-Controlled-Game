extends Area2D

@export var target_level : PackedScene

func _on_body_entered(body):
	

	if body is CharacterBody2D:
		
		get_tree().change_scene_to_packed(target_level)
