GRANT ALL PRIVILEGES ON DATABASE hpc2edge TO hpc2edge;
ALTER DATABASE hpc2edge OWNER TO hpc2edge;
ALTER TABLE edge_measurements OWNER TO hpc2edge;
ALTER TABLE edge_devices OWNER TO hpc2edge;
ALTER TABLE network_architectures OWNER TO hpc2edge;
ALTER TABLE benchmark_result OWNER TO hpc2edge;

-- Essential schema

GRANT SELECT ON TABLE edge_measurements TO hpc2edge_hpc;
GRANT SELECT ON TABLE edge_devices TO hpc2edge_hpc;
GRANT SELECT ON TABLE network_architectures TO hpc2edge_hpc;
GRANT INSERT ON TABLE network_architectures TO hpc2edge_hpc;
GRANT USAGE, SELECT ON SEQUENCE network_architectures_id_seq TO hpc2edge_hpc;

GRANT SELECT ON TABLE edge_measurements TO hpc2edge_edge;
GRANT SELECT ON TABLE edge_devices TO hpc2edge_edge;
GRANT SELECT ON TABLE network_architectures TO hpc2edge_edge;
GRANT INSERT ON TABLE edge_measurements TO hpc2edge_edge;

-- Extended, OpenML-based schema

GRANT SELECT ON TABLE evaluation_measure, task_type, estimation_procedure_type, estimation_procedure, benchmarks, benchmark_result TO hpc2edge_hpc, hpc2edge_edge, hpc2edge;
GRANT INSERT ON TABLE benchmark_result TO hpc2edge_hpc;
GRANT USAGE, SELECT ON SEQUENCE benchmark_result_id_seq TO hpc2edge_hpc;
GRANT USAGE, SELECT ON SEQUENCE edge_measurements_id_seq TO hpc2edge_edge;
