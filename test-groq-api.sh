#!/bin/bash
# Test Groq API with curl
# Usage: bash test-groq-api.sh

# Read API key from .env file
API_KEY=$(grep GROQ_API_KEY .env | cut -d '=' -f2)

echo "Testing Groq API..."
echo ""

curl -X POST https://api.groq.com/openai/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b-8192",
    "messages": [
      {
        "role": "user",
        "content": "Hello! What is 2+2? Please answer briefly."
      }
    ]
  }'

echo ""
echo ""

