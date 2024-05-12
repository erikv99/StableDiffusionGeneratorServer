<h1 align="center" id="title">Stable Diffusion Flask API</h1>

<p id="description">Flask-based RESTful API for Stable Diffusion Image Generation using the [PhotoMaker](https://github.com/TencentARC/PhotoMaker) adapter</p>

<h2>Description</h2> 
A flasked based RESTful API application which is designed to run in a GPU-accelerated (cloud) environment and handle incoming requests for image generation in a queue like order.
The images will be created using a StableDiffusion pipeline and the [PhotoMaker](https://github.com/TencentARC/PhotoMaker) adapter.

Will form a good basis for development tools I plan to create which wil call this application for its generation. 

<h2>💻 Used technologies</h2>

Technologies used in the project:

*   Python
*   Flask
*   Stable Diffusion
*   CUDA
*   Torch
*   [PhotoMaker](https://github.com/TencentARC/PhotoMaker)
