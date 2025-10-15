from googleapiclient import discovery
from google.oauth2 import service_account
import json
import sys

def create_instance(project, zone, instance_file, credentials_file=None):
    # Load instance configuration
    with open(instance_file, 'r') as f:
        instance_config = json.load(f)

    # Authenticate
    if credentials_file:
        credentials = service_account.Credentials.from_service_account_file(credentials_file)
    else:
        credentials = None  # Uses default credentials (e.g., `gcloud auth application-default login`)

    service = discovery.build('compute', 'v1', credentials=credentials)

    # Create the instance
    request = service.instances().insert(
        project=project,
        zone=zone,
        body=instance_config
    )
    response = request.execute()

    print("Instance creation request sent. Operation details:")
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python create_instance.py <PROJECT_ID> <ZONE> <INSTANCE_JSON> [CREDENTIALS_JSON]")
        sys.exit(1)

    project_id = sys.argv[1]
    zone = sys.argv[2]
    instance_json = sys.argv[3]
    credentials_json = sys.argv[4] if len(sys.argv) > 4 else None

    create_instance(project_id, zone, instance_json, credentials_json)
