pipeline {
    environment {
        registry = "Azka200029/i200658_i202478_Mlops_A1" 
        registryCredential = 'docker-hub-credentials' 
        dockerImage = ''
    }
    agent any
    stages {
        stage('Get Dockerfile from GitHub') {
            steps {
                git branch: 'main', url: 'https://github.com/Azka200029/i200658_i202478_Mlops_A1.git' 
            }
        }
        stage('Build Docker image') {
            steps {
                script {
                    dockerImage = docker.build(registry + ":$BUILD_NUMBER")
                }
            }
        }
        stage('Push Docker image to Docker Hub') {
            steps {
                script {
                    docker.withRegistry('', registryCredential) {
                        dockerImage.push()
                    }
                }
            }
        }
    }
}