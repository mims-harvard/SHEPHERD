# Run Exomiser on patient cohort

1) **Generate pedigree files for patients.** Each patient needs to have their own pedigree file. Please adhere to [this format](http://pngu.mgh.harvard.edu/~purcell/plink/data.shtml#ped).

2) **Generate VCF files for patients.** We align patients' whole genome sequencing data to the GRCh38.p13/hg38 human genome build, and perform variant calling using the Genome Analysis Toolkit (GATK) best practices workflow. Please refer to [this paper](https://pubmed.ncbi.nlm.nih.gov/33580225/) for more detail. Note that each VCF file for each patient should include all samples listed in the pedigree. Ensure that the sample IDs correspond exactly between the VCF files and the pedigree files.

3) **Generate parameter file for running Exomiser.** We provide a template (`udn-analysis-exome.yml`) using our selected parameters for Exomiser.

4) **Run Exomiser.** We provide a bash script (`run_exomiser.sh`) with example commands.