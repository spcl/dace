pipeline {
    agent { docker { image 'python:3-stretch' } }
    stages {
        stage('setup') {
            steps {
                sh 'python --version'
            }
        }
    }
}
