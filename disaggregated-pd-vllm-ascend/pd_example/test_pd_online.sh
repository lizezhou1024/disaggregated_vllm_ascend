# serve two example requests
prompt1="San Francisco is a"
prompt2="peking university is the most"
# "model": "/models/DeepSeek-R1-Distill-Qwen-7B",
# Start the vLLM server with the specified model
output1=$(curl -X POST -s http://localhost:9000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "/models/DeepSeek-R1-Distill-Qwen-32B",
"prompt": "'"$prompt1"'",
"max_tokens": 100,
"temperature": 0
}')

output2=$(curl -X POST -s http://localhost:9000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "/models/DeepSeek-R1-Distill-Qwen-32B",
"prompt": "'"$prompt2"'",
"max_tokens": 100,
"temperature": 0
}')

text1=$(echo "$output1" | jq -r '.choices[0].text')
text2=$(echo "$output2" | jq -r '.choices[0].text')


# Print the outputs of the curl requests
echo ""
echo "input of first request: $prompt1"
echo "result of first request: $prompt1$text1"
echo "Output of first request: $output1"

echo "Input of first request: $prompt2"
echo "result of second request: $prompt2$text2"
echo "Output of second request: $output2"
echo "🎉🎉 Successfully finished 2 test requests! 🎉🎉"
echo ""
