package geometry

Shader :: struct {
	vertex:   string,
	fragment: string,
}
Material :: struct {
	color:  [4]f32,
	shader: Shader,
}
