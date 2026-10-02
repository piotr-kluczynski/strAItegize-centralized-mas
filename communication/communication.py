import socket

from strAItegize_common.communication.models import ObserveUnitRequest, ObserveUnitResponse
from strAItegize_common.communication.models import ObserveRegionRequest, ObserveRegionResponse
from strAItegize_common.communication.models import GetUnitsRequest, GetUnitsResponse
from strAItegize_common.communication.models import GetRegionsRequest, GetRegionsResponse
from strAItegize_common.communication.models import GetUnitDetailsRequest, GetUnitDetailsResponse
from strAItegize_common.communication.models import GetRegionDetailsRequest, GetRegionDetailsResponse
from strAItegize_common.communication.models import GetPlayersRequest, GetPlayersResponse
from strAItegize_common.communication.models import IssueOrdersReuqest, IssueOrdersResponse
from strAItegize_common.communication.models import MessagePlayerRequest, MessagePlayerResponse
from strAItegize_common.communication.models import GetPlayerConversationRequest, GetPlayerConversationResponse
from strAItegize_common.communication.models import ErrorResponse

from strAItegize_common.communication.transport import send_message, receive_message
from typing import Literal

class Communication:
    def __init__(self):
        self.simulation_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        self.simulation_port = None

    def start_connection(self, port):
        try:
            self.simulation_socket.connect(("127.0.0.1", port))
            print("Connected to simulation app")

            return True
        except socket.error as e:
            print(f"Connection failed: {e}")
            return False

    def await_simulation_start(self, timeout=30):
        self.simulation_socket.settimeout(timeout)

        # Sending READY status
        send_message(
            self.simulation_socket,
            {
                "type": "READY",
                "payload": {}
            }
        )

        try:
              response = receive_message(self.simulation_socket)
        except socket.timeout:
            print("Waiting for simulation timeout")
            return None

        if response is None:
            return None

        if response.get("type") == "INIT_SIMULATION":
            print("Simulation initialized.")
            return response["payload"]

        elif response["type"] == "ERROR":
            print(response["payload"]["desc"])
        else:
            print(f"Unexpected message: {response.get('type')}")

        return None

    def await_round_start(self, timeout=30):
        self.simulation_socket.settimeout(timeout)

        # Sending READY status
        send_message(
            self.simulation_socket,
            {
                "type": "READY",
                "payload": {}
            }
        )

        try:
            response = receive_message(self.simulation_socket)
        except socket.timeout:
            print("Waiting for simulation timeout")
            return None

        if response is None:
            return None

        if response.get("type") == "INIT_ROUND":
            print("Round initialized.")
            return response["payload"]
        elif response.get("type") == "EXECUTE_UPKEEP":
            print("Upkeep phase initialized. Unit recruitment begins")
            response["payload"]["upkeep_phase"] = True
            return response["payload"]
        elif response.get("type") == "FIN_SESSION":
            print("Received FIN_SESSION signal.")
            self.simulation_socket.close()

            return {"end_session": True}
        elif response["type"] == "ERROR":
            print(response["payload"]["desc"])
        else:
            print(f"Unexpected message: {response.get('type')}")

        return None

    def execute_upkeep(self, recruitment):
        message = {
            "type": "RECRUITMENT_CHOICE",
            "payload": {
                "recruitment": recruitment
            }
        }

        send_message(
            self.simulation_socket,
            message
        )

    def request_data(
            self, 
            request_type=Literal["OBSERVE_UNIT", "OBSERVE_REGION", "GET_UNITS", "GET_REGIONS", "GET_UNIT_DETAILS", "GET_REGION_DETAILS", "GET_PLAYERS", "MESSAGE_PLAYER", "GET_PLAYER_CONVERSATION"], 
            request_args=None
        ):

        # Selecting the request, response model based on the request_type
        request_model = None
        response_model = None

        if request_type == "OBSERVE_UNIT":
            request_model = ObserveUnitRequest
            response_model = ObserveRegionResponse
        elif request_type == "OBSERVE_REGION":
            request_model = ObserveRegionRequest
            response_model = ObserveRegionResponse
        elif request_type == "GET_UNITS":
            request_model = GetUnitsRequest
            response_model = GetUnitsResponse
        elif request_type == "GET_REGIONS":
            request_model = GetRegionsRequest
            response_model = GetRegionsResponse
        elif request_type == "GET_UNIT_DETAILS":
            request_model = GetUnitDetailsRequest
            response_model = GetUnitDetailsResponse
        elif request_type == "GET_REGION_DETAILS":
            request_model = GetRegionDetailsRequest
            response_model = GetRegionDetailsResponse
        elif request_type == "GET_PLAYERS":
            request_model = GetPlayersRequest
            response_model = GetPlayersResponse     
        elif request_type == "MESSAGE_PLAYER":
            request_model = MessagePlayerRequest
            response_model = MessagePlayerResponse        
        elif request_type == "GET_PLAYER_CONVERSATION":
            request_model = GetPlayerConversationRequest
            response_model = GetPlayerConversationResponse   
        else:
            return None 
        
        # Creating the payload
        message = {
            "type": request_type,
            "payload": {}
        }

        if request_args is not None:
            request_data = request_model(**request_args)
            message["payload"] = request_data.model_dump()

        # Sending a request
        send_message(
            self.simulation_socket,
            message
        )

        # Receiving and parsing the response
        raw_response = receive_message(self.simulation_socket)

        if not raw_response or raw_response.get("type") == "ERROR":
            return None

        return response_model.model_validate(raw_response["payload"])

    def end_round(self, orders):
        message = {
            "type": "ISSUE_ORDERS",
            "payload": {
                "orders": orders
            }
        }

        send_message(
            self.simulation_socket,
            message
        )

        response = receive_message(self.simulation_socket)

        if response is None:
            return False

        if response.get("type") == "FIN_ROUND":
            print("Round ended.")
            return True
        else:
            print(f"Unexpected message: {response.get("type")}")
            return False