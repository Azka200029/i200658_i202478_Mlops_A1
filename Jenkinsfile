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
                bat 'docker build -t docker_image .'
            }
        }

        stage('Push to Docker Hub') {
            steps {
                bat 'docker login'
                bat 'docker tag docker_image Azka200029/i200658_i202478_Mlops_A1:first_tag'
                bat 'docker push Azka200029/i200658_i202478_Mlops_A1:first_tag'
            }
        }
    }
}