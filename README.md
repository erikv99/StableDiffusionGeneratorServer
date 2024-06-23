<h1 align="center" id="title">Stable Diffusion Flask API</h1>

<p id="description">
  Flask-based RESTful API for Stable Diffusion Image img2img using the 
  <a href="https://github.com/TencentARC/PhotoMaker">PhotoMaker</a> adapter
</p>

<h2>Instruction list</h2>
I Use currently use hypercloud for my setup.

1. setup vm with public ip
2. enable ssh connection
3. make sure the vm firewall is configured to allow incoming on the port the app is hosted on. (and outgoing ofc)
4. connect in vs code 
5. open new terminal
6. sudo apt update
7. git clone https://github.com/erikv99/StableDiffusionGeneratorServer.git
8. cd StableDiffusionGeneratorServer
9. git clone https://github.com/TencentARC/PhotoMaker.git
10 sudo apt install -y python3-pip
11. sudo apt install -y python3-venv
12. python3 -m venv env
13. source env/bin/activate
14. pip install -r requirements.txt
15. Run the application

<h2>Description</h2> 
A flasked based RESTful API application which is designed to run in a GPU-accelerated (cloud) environment and handle incoming requests for image generation in a queue-like order.
The images will be created using a StableDiffusion pipeline and the 
<a href="https://github.com/TencentARC/PhotoMaker">PhotoMaker</a> adapter.

Will form a good basis for development tools I plan to create which will call this application for its generation. 

<h2>Brief rundown</h2>
Will take 1 or multiple images of a person and combine these with a prompt to place said person in a different context.

<h2>💻 Used technologies</h2>

Technologies used in the project:

<ul>
  <li>Python</li>
  <li>Flask</li>
  <li>Stable Diffusion</li>
  <li>CUDA</li>
  <li>Torch</li>
  <li><a href="https://github.com/TencentARC/PhotoMaker">PhotoMaker</a></li>
</ul>

