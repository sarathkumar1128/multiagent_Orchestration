# Entry point for the multi-agent CLI application
from logger_config import setup_logger
from coordinator import Coordinator

def main():
    setup_logger()

    user_input = "Create a production-ready Task Management Application frontend react,middleware python,backend mysqls."

    coordinator = Coordinator()
    output = coordinator.execute(user_input, session_id="default_session")

    print("\n\n=========== GENERATED APPLICATION ===========\n")
    print(output)


if __name__ == "__main__":
    main()
