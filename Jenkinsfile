pipeline {
    environment {
        registry = "azkaasim/i200658_i202478_mlop_a1" 
        registryCredential = 'credentials-docker' 
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

// pipeline {
//     environment {
//         registry = "azkaasim/i200658_i202478_mlop_a1" 
//         registryCredential = 'docker-credentials' 
//         dockerImage = ''
//     }
//     agent any
//     stages {
//         stage('Get Dockerfile from GitHub') {
//             steps {
//                 git branch: 'main', url: 'https://github.com/Azka200029/i200658_i202478_Mlops_A1.git' 
//             }
//         }
//         stage('Build Docker image') {
//             steps {
//                 script {
//                     dockerImage = docker.build(registry + ":$BUILD_NUMBER")
//                 }
//             }
//         }
//         stage('Push Docker image to Docker Hub') {
//             steps {
//                 script {
//                     docker.withRegistry('', registryCredential) {
//                         dockerImage.push()
//                     }
//                 }
//             }
//         }
//         stage('Send Email Notification') {
//             when {
//                 branch 'main' 
//             }
//             steps {
//                 script {
//                     // Sending email notification upon successful build
//                     emailext (
//                         to: 'i202478@nu.edu.pk', 
//                         subject: "Merged to main branch successfully",
//                         body: "The merge to the main branch was successful.",
//                         attachLog: true,
//                         mimeType: 'text/plain'
//                     )
//                 }
//             }
//         }
//     }
// }
