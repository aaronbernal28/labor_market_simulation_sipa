rule extract_raw:
    """Unpack MLER RAR archive to data/raw/."""
    input:
        archive=config["datasets"]["sipa"]["source"],
    output:
        csv="data/raw/mler.csv",
    log:
        "logs/extract_raw.log",
    shell:
        "unrar e {input.archive} data/raw/ > {log} 2>&1"
