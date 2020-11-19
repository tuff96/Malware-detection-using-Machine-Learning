# Malware-detection-using-Machine-Learning
The scope of this paper is to present a malware detection approach using machine learning. In this paper we will focus on windows executable ﬁles. Because of the abnormal growth of these malicious software’s we need to use diﬀerent automated approaches to ﬁnd theses infected ﬁles. 

In this project we are going to study and implement a script used for data extraction from the PE-ﬁles to create a data set with infected and clean ﬁles, on which we are gonna train our machine learning algorithms:K-nn, XGBoost and Random Forest. 

The last chapter of this paper the algorithms are tested with all the data set features. The accuracy of all algorithms is over 90%. After applying a Feature selection algorithm over the data set, the accuracy has been improved for all the learning algorithms

EDIT:
PE files: https://en.wikipedia.org/wiki/Portable_Executable

In the legitimate folder, you need to add a lot of legitimate windows PE files (just download random PE files from an  legitimate source like skype.exe, teams.exe, etc)
and " /hdd/Downloads/virusi_00325/" is a folder full of Malwares from https://virusshare.com/ (send them an email and ask for an account explaining why you need one )

