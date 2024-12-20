"""Simple script to test and compare boto3 and LangChain requests to AWS Bedrock."""

import http.client
import json
import logging
import os

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

# Enable HTTP debugging
http.client.HTTPConnection.debuglevel = 1

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

logger = logging.getLogger(__name__)


def log_langchain_request(method, url, headers, body):
    """Explicitly log LangChain request details."""
    print("\n--- LANGCHAIN RAW REQUEST ---")
    print(f"Method: {method}")
    print(f"URL: {url}")
    print("Headers:")
    for key, value in headers.items():
        print(f"  {key}: {value}")
    print("Body:")
    print(json.dumps(body, indent=2))
    print("--- END LANGCHAIN RAW REQUEST ---\n")


def main():
    """Run test requests with both standard and optimized configurations."""
    load_dotenv(override=True)

    # Test query
    query = "What is the capital of France? Keep the answer very short."
    model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

    logger.info("Setting up clients...")

    # Set up boto3 client
    boto3_client = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION"),
        config=Config(
            parameter_validation=False,  # This allows us to see raw requests
        ),
    )

    # Monkey patch boto3 client's _make_api_call to log requests
    original_make_api_call = boto3_client._make_api_call

    def logging_make_api_call(self, operation_name, api_params):
        print("\n--- BOTO3 RAW REQUEST ---")
        print(f"Operation: {operation_name}")
        print("Parameters:")
        print(json.dumps(api_params, indent=2))
        print("--- END BOTO3 RAW REQUEST ---\n")
        return original_make_api_call(operation_name, api_params)

    boto3_client._make_api_call = lambda *args, **kwargs: logging_make_api_call(boto3_client, *args, **kwargs)

    # Test configurations
    configs = [
        # "standard",
        "optimized",
    ]

    logger.info("\nTesting LangChain requests:")
    for config in configs:
        logger.info(f"\nMaking LangChain request with {config} config...")

        # Create a custom client that logs the request before sending
        class LoggingChatBedrockConverse(ChatBedrockConverse):
            def _prepare_request(self, messages):
                # Call the original method to prepare the request
                prepared_request = super()._prepare_request(messages)

                # Log the request details
                log_langchain_request(
                    method=prepared_request.method,
                    url=prepared_request.url,
                    headers=dict(prepared_request.headers),
                    body=json.loads(prepared_request.body.decode("utf-8")),
                )

                return prepared_request

        client = LoggingChatBedrockConverse(
            model_id=model_id,
            region_name=os.getenv("AWS_REGION"),
            performance_config={"latency": config},
        )
        response = client.invoke(query)
        print(f"LangChain Response: {response.content}")

    logger.info("\nTesting boto3 requests:")
    for config in configs:
        logger.info(f"\nMaking boto3 request with {config} config...")
        message = {"role": "user", "content": [{"text": query}]}
        response = boto3_client.converse(
            modelId=model_id,
            messages=[message],
            performanceConfig={"latency": config},
        )
        response_text = response["output"]["message"]["content"][0]["text"]
        print(f"Boto3 Response: {response_text}")


if __name__ == "__main__":
    main()
