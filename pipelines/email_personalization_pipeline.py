# pipelines/email_personalization_pipeline.py

import openai
import os
import logging
from dotenv import load_dotenv

# Initialize logging
logging.basicConfig(
    filename="../data/logs/email_personalization.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Personalize email
def personalize_email(template, customer_name, product_details):
    try:
        prompt = f"Personalize the following email template for a customer named {customer_name} interested in {product_details}:\n\n{template}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in personalizing email content.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.7,
            n=1,
        )
        personalized_email = response.choices[0].message["content"].strip()
        logging.info(f"Personalized email generated for customer: {customer_name}")
        return personalized_email
    except Exception as e:
        logging.error(f"Error personalizing email: {e}")
        raise e


# Example usage
if __name__ == "__main__":
    email_template = """
    Dear Customer,

    We are excited to introduce our latest product that we believe will greatly benefit your business. Our new solution offers unparalleled features designed to streamline your operations and enhance productivity.

    Best regards,
    Your Company
    """
    print(
        personalize_email(
            email_template, "John Doe", "our latest productivity solution"
        )
    )
