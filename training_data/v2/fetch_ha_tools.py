
import requests
import json
import os

HA_URL = "http://homeassistant.local:8123"
HA_TOKEN = os.environ.get("HA_ACCESS_TOKEN")

headers = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json",
}

def fetch_ha_data():
    # Fetch states
    try:
        states_response = requests.get(f"{HA_URL}/api/states", headers=headers)
        states_response.raise_for_status()
        with open("/mnt/zardos/charm-hal-env/training_data/v2/ha_states.json", "w") as f:
            json.dump(states_response.json(), f, indent=2)
        print("HA states fetched and saved.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching HA states: {e}")

    # Fetch services
    try:
        services_response = requests.get(f"{HA_URL}/api/services", headers=headers)
        services_response.raise_for_status()
        # The plan mentions saving ha_tools.json, but the description for fetching tools
        # suggests scraping the conversation API logs. For now, I will save the services
        # as a starting point for ha_tools.json, as it contains a lot of the tool definitions.
        with open("/mnt/zardos/charm-hal-env/training_data/v2/ha_tools.json", "w") as f:
            json.dump(services_response.json(), f, indent=2)
        print("HA services fetched and saved as ha_tools.json.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching HA services: {e}")

    # For conversation/Assist tool list, the plan suggests enabling logger debug and scraping logs.
    # This is more complex to automate directly via a script. 
    # I will proceed with the states and services for now, and if the user wants to proceed
    # with the log scraping, I will address it in a later step.

if __name__ == "__main__":
    fetch_ha_data()
