release: shader
	odin run . -out:bin/hello-vk
debug: shader
	odin run . -out:bin/hello-vk -debug
shader: rendering/shaders/shader.vert rendering/shaders/shader.frag
	glslc rendering/shaders/shader.vert -o rendering/shaders/vert.spv
	glslc rendering/shaders/shader.frag -o rendering/shaders/frag.spv
