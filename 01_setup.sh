#!/bin/bash

# Always start the SSH agent
eval $(ssh-agent -s)

# Remove any existing SSH key pair in the ~/.ssh directory
rm -rf ~/.ssh/id_rsa ~/.ssh/id_rsa.pub

# Hardcoded GitHub email
email="zvonimirsucic.dl@gmail.com"
echo "Using email $email"

# Generate an SSH key pair with the hardcoded email
ssh-keygen -t rsa -b 4096 -C "$email"

# Add the generated private key to the SSH agent
ssh-add ~/.ssh/id_rsa

# Read the generated public key
pub=$(cat ~/.ssh/id_rsa.pub)

# Hardcoded GitHub username
githubuser="zsucicdl"
echo "Using username $githubuser"

# Prompt only for the personal access token
read -p "Enter GitHub Personal Access Token for user $githubuser: " githubtoken
echo  # Add a newline for cleaner output after the silent read command

# Use the curl command to send a POST request to the GitHub API to add the SSH key to the user's GitHub account
response=$(curl -u "$githubuser:$githubtoken" -X POST -H "Content-Type: application/json" -d "{\"title\":\"`hostname`\",\"key\":\"$pub\"}" https://api.github.com/user/keys)

# Check and notify the user if the key addition was successful
if [[ $response == *"key"* ]]; then
  echo "SSH key added successfully!"

  # Wait for 30 seconds to ensure the key is propagated on the GitHub side
  echo "Waiting for 30 seconds before attempting to clone..."
  sleep 30

  # Perform the git clone operation
  git clone git@github.com:zsucicdl/llm.git
else
  echo "Failed to add SSH key. Please check your GitHub credentials and try again."
fi