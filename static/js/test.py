           print(f"User input: {user_input}")
            messages = [
            {
                "role": "system",
                "content": s_p,
            },
            {"role": "user", "content": user_input},
            ]
