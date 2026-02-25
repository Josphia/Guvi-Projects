import mlflow

# 1. Use a simple relative path string (no 'file://' prefix)
# This is the most "Pythonic" way to handle Windows paths in MLflow
mlflow.set_tracking_uri("mlruns")

# 2. Set the experiment name
mlflow.set_experiment("Simple_Python_Experiment")

print(f"Tracking URI set to: {mlflow.get_tracking_uri()}")

# 3. Start a test run
with mlflow.start_run(run_name="Local_Folder_Test"):
    print("Logging data to local folder...")
    mlflow.log_param("mode", "local_folder")
    mlflow.log_metric("success_rate", 1.0)
    
print("Run complete! Now check the UI.")