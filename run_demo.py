import subprocess
import time
import sys

def print_header(text):
    print(f"\n{'='*50}")
    print(f"🚀 {text.upper()}")
    print(f"{'='*50}\n")

def run_script(script_name, description):
    print(f"⏳ Executing {description} ({script_name})...")
    time.sleep(1) # Dramatic pause for the demo
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✅ {script_name} completed successfully.\n")
        time.sleep(1)
    except subprocess.CalledProcessError:
        print(f"\n❌ ERROR: {script_name} failed. Check the logs.")
        sys.exit(1)

if __name__ == "__main__":
    print("\n" + "*"*60)
    print("   NODEBIAS ENTERPRISE AUDIT: DEMO SEQUENCE INITIATED")
    print("*"*60)

    # Act 0: Ensure models are frozen
    run_script('freeze_models.py', "Model Serialization")

    # Act 1: Data Bias
    print_header("ACT 1: Auditing Historical Data Bias")
    run_script('data_engine.py', "Baseline Data Engine")

    # Act 2: Model Bias
    print_header("ACT 2: Auditing AI Amplification")
    run_script('model_engine.py', "Logistic Regression Auditor")
    run_script('model_engine_v2.py', "Naive Bayes Auditor")

    # Act 3: Handoff to UI
    print_header("ACT 3: Mitigation Gateway Ready")
    print("All backend audits complete and JSON reports generated.")
    print("Run `python app.py` and open the React dashboard to apply mitigation.")
    print("\n" + "*"*60 + "\n")
