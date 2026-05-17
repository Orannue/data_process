python generate_prompts_via_api.py \
  --base_url "https://api.xi-ai.cn/v1" \
  --api_key "sk-qsIWOY8jWlPjW8Qb9e138e2a33384f848a82Fa394eF8E4Fe" \
  --model "gemini-3-pro" \
  --number_of_shots 4 \
  --total_characters 3 \
  --max_characters_per_shot 2 \
  --num_generations 10 \
  --output "eval/generated_prompts_new.json"