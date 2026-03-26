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
                            docker run --rm -v "$WORKSPACE/outputs:/app/outputs" "${IMAGE_NAME}"
                        '''
                    } else {
                        bat '''
                            if not exist outputs\\metrics mkdir outputs\\metrics
                            if not exist outputs\\plots mkdir outputs\\plots
                            docker run --rm -v "%WORKSPACE%\\outputs:/app/outputs" %IMAGE_NAME%
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
