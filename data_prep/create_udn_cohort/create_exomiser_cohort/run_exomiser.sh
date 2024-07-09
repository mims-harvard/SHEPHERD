#! /bin/bash
#SBATCH --mem=5G
#SBATCH -c 1
#SBATCH -t 10:00:00
#SBATCH -p short 
#SBATCH -o exomiser.out


module load java

# for FILE in udn_analysis_exome_ymls/*; do sbatch run_exomiser.sh $FILE; done
#java -Xms2g -Xmx4g -jar exomiser-cli-13.0.1.jar --analysis ${1}


# for FILE in udn_analysis_exome_ymls/batch*; do sbatch run_exomiser.sh $FILE; done
java -Xms2g -Xmx4g -jar exomiser-cli-13.0.1.jar --analysis-batch ${1}

