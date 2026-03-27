pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        IMAGE_NAME = 'sdss-ml-pipeline'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                script {
                    if (isUnix()) {
                        sh 'pip install -r requirements.txt'
                    } else {
                        bat 'pip install -r requirements.txt'
                    }
                }
            }
        }

        stage('Basic Dataset Tests') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            python -c "
                            import pandas as pd
                            import sys

                            df = pd.read_csv('sdss_sample.csv')
                            required_cols = ['u', 'g', 'r', 'i', 'z', 'redshift', 'class']
                            missing = [c for c in required_cols if c not in df.columns]
                            if missing:
                                print(f'Missing columns: {missing}')
                                sys.exit(1)
                            if df.empty:
                                print('Dataset is empty')
                                sys.exit(1)
                            print(f'Dataset OK: {df.shape[0]} rows, {df.shape[1]} columns')
                            print(f'Classes: {df[\"class\"].unique().tolist()}')
                            "
                        '''
                    } else {
                        bat '''
                            python -c "import pandas as pd; df=pd.read_csv('sdss_sample.csv'); required=['u','g','r','i','z','redshift','class']; missing=[c for c in required if c not in df.columns]; print('Missing: '+str(missing)) if missing else print('OK: '+str(df.shape))"
                        '''
                    }
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            docker build -t "${IMAGE_NAME}" .
                        '''
                    } else {
                        bat '''
                            docker build -t %IMAGE_NAME% .
                        '''
                    }
                }
            }
        }

        stage('Run Pipeline In Docker') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            mkdir -p outputs/metrics outputs/plots
                            CONTAINER_NAME="sdss-ml-pipeline-${BUILD_NUMBER}"
                            docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
                            docker create --name "$CONTAINER_NAME" "${IMAGE_NAME}" >/dev/null
                            docker start -a "$CONTAINER_NAME"
                            docker cp "$CONTAINER_NAME:/app/outputs/." outputs/
                            docker rm -f "$CONTAINER_NAME"
                        '''
                    } else {
                        bat '''
                            if not exist outputs\\metrics mkdir outputs\\metrics
                            if not exist outputs\\plots mkdir outputs\\plots
                            set CONTAINER_NAME=sdss-ml-pipeline-%BUILD_NUMBER%
                            docker rm -f %CONTAINER_NAME% >nul 2>nul
                            docker create --name %CONTAINER_NAME% %IMAGE_NAME% >nul
                            docker start -a %CONTAINER_NAME%
                            docker cp %CONTAINER_NAME%:/app/outputs/. outputs/
                            docker rm -f %CONTAINER_NAME%
                        '''
                    }
                }
            }
        }

        stage('Validate Outputs') {
            steps {
                script {
                    def requiredArtifacts = [
                        'outputs/metrics/pipeline_report.json',
                        'outputs/metrics/summary.txt',
                        'outputs/metrics/classification_metrics.json',
                        'outputs/metrics/regression_metrics.json',
                        'outputs/metrics/clustering_metrics.json',
                        'outputs/plots/classification_confusion_matrix.png',
                        'outputs/plots/regression_actual_vs_predicted.png',
                        'outputs/plots/clustering_projection.png',
                        'outputs/plots/clustering_vs_class.png',
                    ]

                    for (artifact in requiredArtifacts) {
                        if (!fileExists(artifact)) {
                            error("Missing expected artifact: ${artifact}")
                        }
                    }
                }
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: 'outputs/**/*', fingerprint: true
            }
        }
    }

    post {
        always {
            echo 'Jenkins pipeline finished.'
        }
        success {
            echo 'Docker image was built, the pipeline ran in a container, and artifacts were archived.'
        }
        failure {
            echo 'Pipeline failed. Review the stage logs to identify the failing step.'
        }
    }
}
