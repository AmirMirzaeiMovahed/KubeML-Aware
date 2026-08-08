from scheduler.execution import parse_execution_marker

logs = """{"config":{"C":10,"G":64.0,"M":256,"P":1,"R":0.5,"T":60.0,"job_id":"workload-1","seed":228711551},"event":"JOB_CONFIG","job_id":"workload-1","timestamp":1786195378.4579875}
{"blas_runtime":{"expected_threads":1,"libraries":[{"architecture":"Haswell","internal_api":"openblas","num_threads":1,"prefix":"libscipy_openblas","threading_layer":"pthreads","user_api":"blas","version":"0.3.30"}]},"blas_threads":1,"estimated_T":60.0,"estimated_T_error_seconds":59.7632,"event":"INITIALIZATION_COMPLETED","job_id":"workload-1","matrix_shape":[256,256],"partition_rows":[[0,256]],"seed":228711551,"timestamp":1786195378.8362923,"work_model":{"checkpoint_bytes":262144,"checkpoint_count":0,"checkpoint_seconds":0.0,"convergence_steps":8,"estimated_training_seconds":0.2368,"gradient_bytes":67108864,"gradient_update_seconds":0.2048,"matrix_compute_seconds":0.032,"model_version":"2.0","partition_sync_seconds":0.0,"planned_steps":8,"step_budget":237,"termination_reason":"converged"}}
{"event":"EXECUTION_STARTED","job_id":"workload-1","timestamp":1786195378.8365655}
{"event":"CONVERGED","job_id":"workload-1","loss":0.01831563888873418,"step":8,"timestamp":1786195379.0208626}
{"blas_library_count":1,"blas_threads":1,"checkpoint_bytes":262144,"checkpoint_count":0,"checkpoint_seconds":0.0,"convergence_steps":8,"duration_seconds":0.18464422225952148,"event":"EXECUTION_COMPLETED","final_loss":0.01831563888873418,"gradient_bytes":67108864,"job_id":"workload-1","step_budget":237,"steps":8,"termination_reason":"converged","timestamp":1786195379.0212097,"work_model_version":"2.0"}"""

result = parse_execution_marker(logs, expected_job_id="workload-1")
print(f"Result: {result}")
