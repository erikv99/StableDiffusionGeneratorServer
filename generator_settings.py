class GeneratorSettings:

    def __init__(self, prompt, negative_prompt, guidance_scale, style_strength, number_of_steps):
    
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.guidance_scale = guidance_scale
        self.style_strength = style_strength
        self.number_of_steps = number_of_steps