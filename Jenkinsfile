pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/Azka200029/i200658_i202478_Mlops_A1.git'
            }
        }

        stage('Build Image') {
            steps {
                bat 'docker build -t docker_image_A1 .'
            }
        }

        stage('Push to Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'azkaasim', passwordVariable: '#ANARAMSmandi292000', usernameVariable: 'azkaasim')]) {
                    sh "echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin"
                    sh "docker build -t your-image-name ."
                    sh "docker push your-username/your-repo-name:your-tag"
            }
        }
    }
}
