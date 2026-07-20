import subprocess
import sys


def run_script(script_path):
    try:
        subprocess.run(
            [sys.executable, script_path],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("\nThe selected operation failed.")
        print("Check the error shown above.")


def show_menu():
    print("\n" + "=" * 55)
    print("INTELLIGENT RESUME-JD MATCHING SYSTEM")
    print("Developed by Gurbani Panesar")
    print("=" * 55)

    print("1. Generate Dataset")
    print("2. Preprocess Dataset")
    print("3. Build and View Model")
    print("4. Train Model")
    print("5. Evaluate Model")
    print("6. Predict Resume-JD Match")
    print("7. Exit")


def main():
    while True:
        show_menu()

        choice = input(
            "\nEnter your choice: "
        ).strip()

        if choice == "1":
            run_script(
                "src/generate_dataset.py"
            )

        elif choice == "2":
            run_script(
                "src/preprocessing.py"
            )

        elif choice == "3":
            run_script(
                "src/model.py"
            )

        elif choice == "4":
            run_script(
                "src/train.py"
            )

        elif choice == "5":
            run_script(
                "src/evaluate.py"
            )

        elif choice == "6":
            run_script(
                "src/predict.py"
            )

        elif choice == "7":
            print(
                "\nProgram closed successfully."
            )
            break

        else:
            print(
                "\nInvalid choice. Enter a number from 1 to 7."
            )


if __name__ == "__main__":
    main()