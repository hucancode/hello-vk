package world
import g "../geometry"

NodeVariant :: union {
	g.Mesh,
	Light,
}
Node :: struct {
	position: [3]f32,
	children: []Node,
	variant:  NodeVariant,
}
