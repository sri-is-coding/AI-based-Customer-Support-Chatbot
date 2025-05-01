# DIA cw Chatbot Srivarshini Selvaraj 20512874
# chatbotbank.py

!pip install -q bitsandbytes transformers accelerate

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

model_path = "/content/drive/MyDrive/zephyr-finetuned-bankbot"

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True  # allows offloading to CPU
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",  # automatically places across GPU/CPU
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

def openchat_assistant():
    print("\nChatbot: You want to type your own question.")
    print("\nChatbot: Type your question in the box below. Type 'exit' and enter to return back to the main menu.")
    chat_history = []

    while True:
        user_query = input("You: ").strip()

        if user_query.lower() in ["exit", "quit", "0"]:
            print("Chatbot: Alright, taking you back to the main menu!")
            break

        short_replies = {
            "hi": "Hello there! How can I assist you today?",
            "hello": "Hi there! How may I help you with your banking needs?",
            "thanks": "You're most welcome! Let me know if you need anything else.",
            "thank you": "Glad to help! Let me know if you need anything else.",
            "okay": "Alright! Let me know if you have another question.",
            "cool": "Awesome! Do you have any other questions?",
            "yes": "Ask your question!",
            "nope": "Awesome, goodbye!",
            "no": "Awesome, goodbye!",
            "great": "Happy to hear that!",
            "bye": "Goodbye! Have a great day ahead.",
            "goodbye": "Take care! Thank you for chatting with us."
        }

        normalized_query = user_query.lower().strip()
        if normalized_query in short_replies:
            response = short_replies[normalized_query]
            print(f"Chatbot: {response}")
            chat_history.append({"user": user_query, "bot": response})
            continue

        context = ""
        for turn in chat_history[-3:]:
            context += f"User: {turn['user']}\nBot: {turn['bot']}\n"
        prompt = f"{context}User: {user_query}\nBot:"

        #inputs = tokenizer(prompt, return_tensors="pt").to(device)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device).to(torch.long) 


        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=80,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response = decoded.split("User:")[-1].split("Bot:")[-1].strip()

        if not response or response.strip().lower() == user_query.strip().lower():
            response = "Hmm, I'm not sure how to answer that just yet. Could you try asking another way?"

        print(f"Chatbot: {response}")

        chat_history.append({
            "user": user_query,
            "bot": response
        })

def chatbot():
    print("\n--------------------------------------------------------")
    print("-----------   SRI's INTERNATIONAL BANK  ----------------")
    print("--------------------------------------------------------")
    print("\nChatbot: Welcome to Sri's International Bank! How can I assist you today?")

    while True:
        print("\nPlease select one of the categories related to your query, or type your own question:") # menu nav

        categories = {
            "1": "My Account",
            "2": "Transaction Support",
            "3": "Loan & Credit Services",
            "4": "Mobile & Online banking",
            "5": "Security & Fraud Prevention",
            "6": "Investment & Savings",
            "7": "Debit Card Services",
            "8": "I want to type my own question",
            "0": "I would like to exit the chat."
        }

        for key, value in categories.items():
            print(f"{key}. {value}")

        choice = input("\nPlease enter the number of your choice: ").strip()

        # if the user types their question instead of number
        if not choice.isdigit():
            print("\nChatbot: Invalid choice. Please try again.")
            continue

        # if the user enters 0 to exit
        if choice == "0":
            print("\nChatbot: Thank you for using our service! Have a great day!")
            print("--------------------------------------------------------")
            break

        elif choice == "8":
            openchat_assistant()

        elif choice in categories:
            selected_category = categories[choice]
            print(f"\nChatbot: You selected: {selected_category}")

            FAQs = {
                "My Account": ["Check account balance", "Update personal details", "Account closure procedure", "Open a new account", "Joint account process"],
                "Transaction Support": ["Failed transaction", "Payment dispute", "International wire transfer", "Exchange rates info", "Using card abroad"],
                "Loan & Credit Services": ["Check loan eligibility", "Credit card details"],
                "Mobile & Online banking": ["Reset internet banking password", "Register for online banking", "Mobile app not working"],
                "Security & Fraud Prevention": ["Lost my card", "Report fraud activity", "Replace lost card"],
                "Investment & Savings": ["Interest rates", "Savings account details", "Open fixed deposit", "Premature withdrawal", "Interest rates for FD"],
                "Debit Card Services": ["Apply for debit card", "Activate debit card", "Replace lost card"]
            }

            for i, question in enumerate(FAQs[selected_category], start=1):
                print(f"{i}. {question}")
            print("0. Type my own question")

            faq_choice = input("\nSelect a question or enter 0 to type your own: ").strip()

            responses = {
                "Check account balance": "You can check your account balance via online banking or our mobile app.",
                "Update personal details": "To update your details, visit the nearest branch or update via internet banking.",
                "Open a new account": "To open a new account, visit our website or your nearest branch with valid ID and address proof.",
                "Joint account process": "Joint accounts can be opened by filling out a joint application form at your branch along with both parties' documents.",
                "Account closure procedure": "Visit your nearest branch and fill out the account closure form. Ensure your balance is zero.",
                "Failed transaction": "If a transaction failed but the amount was deducted, it should be refunded in 24-48 hours.",
                "Payment dispute": "For disputes, submit a claim through your banking app under 'Transaction Disputes'.",
                "International wire transfer": "You can initiate an international transfer via online banking. You’ll need the recipient’s SWIFT code and IBAN.",
                "Using card abroad": "Enable international usage in your mobile app under 'Card Settings'. Charges may apply based on location.",
                "Exchange rates info": "Check real-time exchange rates in our mobile app or website under the 'Forex Rates' section.",
                "Check loan eligibility": "Check your loan eligibility by logging into your bank profile and navigating to 'Loan Offers'.",
                "Credit card details": "For your credit card limit, billing date, and offers, visit the credit card section in our app.",
                "Reset internet banking password": "To reset your password, go to the login page and click on 'Forgot Password'. Follow the steps to verify your identity.",
                "Register for online banking": "You can register by visiting our online portal and clicking on 'New User Registration'. Have your account number ready.",
                "Mobile app not working": "Please ensure your app is updated. If the issue persists, reinstall the app or contact support.",
                "Lost my card": "Please block your card immediately via the app or call customer support to prevent misuse.",
                "Report fraud activity": "If you suspect fraud, call our 24/7 fraud helpline or visit the nearest branch.",
                "Interest rates": "Check current interest rates on deposits and loans via our bank's website or mobile app.",
                "Savings account details": "For opening or managing savings accounts, visit our online banking portal.",
                "Open fixed deposit": "You can open a fixed deposit from the app by navigating to 'Deposits' > 'Open New FD'.",
                "Premature withdrawal": "Premature withdrawal may incur a penalty. Please refer to the terms or visit your branch for assistance.",
                "Interest rates for FD": "Interest rates vary based on tenure and amount. Visit our website for the latest FD interest rate table.",
                "Apply for debit card": "You can apply for a new debit card through our mobile app under 'Card Services' or by visiting your nearest branch.",
                "Activate debit card": "To activate your debit card, call the activation number provided or use the 'Activate Card' feature in your mobile banking app.",
                "Replace lost card": "If your card is lost, block it immediately and apply for a replacement through our app or by calling customer care."
            }

            if faq_choice == "0":
                openchat_assistant()
            elif faq_choice.isdigit() and int(faq_choice) <= len(FAQs[selected_category]):
                selected_question = FAQs[selected_category][int(faq_choice) - 1]
                print(f"\nChatbot: You asked: {selected_question}")
                print(f"Chatbot: {responses[selected_question]}")
                print("\nChatbot: Do you have any other questions?")
            else:
                print("\nChatbot: Invalid choice. Please try again.")

        else:
            print("\nChatbot: Invalid input. Please select a number from the list.")

chatbot()
