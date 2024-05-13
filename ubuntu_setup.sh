#!/bin/bash

# Setting it so it exits on failure.
set -e

# Update package list and install pip and venv
echo "Updating package list..."
sudo apt update

echo "Installing pip..."
sudo apt install -y python3-pip

echo "Installing Python venv..."
sudo apt install -y python3-venv

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv env

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate

# Install requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Configure git
echo "Configuring git..."
read -p "Enter your git username: " username
read -p "Enter your git email: " email
git config --global user.name "$username"
git config --global user.email "$email"

# Install PhotoMaker package
echo "Installing PhotoMaker package..."
pip install git+https://github.com/TencentARC/PhotoMaker.git

# Clone PhotoMaker repository
echo "Cloning PhotoMaker repository..."
git clone https://github.com/TencentARC/PhotoMaker.git

echo "Setup complete."

echo "To make this script executable, run: chmod +x ubuntu_setup.sh"
echo "To execute the script, run: ./ubuntu_setup.sh"

