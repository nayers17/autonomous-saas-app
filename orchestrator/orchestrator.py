# orchestrator.py

from pipelines.content_generation_pipeline import generate_content
from pipelines.customer_segmentation_pipeline import segment_customers
from pipelines.ad_optimization_pipeline import optimize_ads
from pipelines.lead_generation_pipeline import generate_leads

# Import other pipelines as needed


def process_request(task_type, input_data):
    if task_type == "content_generation":
        return generate_content(input_data)
    elif task_type == "customer_segmentation":
        return segment_customers(input_data)
    elif task_type == "ad_optimization":
        return optimize_ads(input_data)
    elif task_type == "lead_generation":
        return generate_leads(input_data)
    else:
        return "Task not recognized."


# Example usage
if __name__ == "__main__":
    # Sample input for customer segmentation
    task = "customer_segmentation"
    sample_data = {
        "customer_data": [
            "Enjoys hiking and nature.",
            "Loves watching movies and reading books.",
        ]
    }

    result = process_request(task, sample_data)
    print(result)
