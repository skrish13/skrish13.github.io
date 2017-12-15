---
layout: post
title: Multimodality in Emotion understanding (in videos) - Part 1
excerpt: "Writing down initial thoughts and approach, critique on current ML objectives and  thoughts on refining the goal"
categories: [multimodality, research]
comments: True
---

I was interested in doing a project which involved multimodality, DL and emotion detection/classification was the one which was easily available and also sounded interesting too. This post is about writing down my initial thoughts and approach, some amount of explanation as to what current ML models lack and providing a better way of thinking to approach the problem. This was written in hindsight, contrary to the timestamp by explaining and reflecting on the ideas formed at that time.

### Initial approach

So, when I was started out, my thinking was very trivial. I had searched for papers which had worked on multimodal emotion recognition using machine learning/deep learning. The idea was to do make a tool by implementing such methods. Most of the papers were trained and tested on few datasets. Examples of such datasets which people work on for emotion recognition in videos:

- EmotiW - Emotion Recognition in the Wild
- MEC - Multimodal Emotion Recognition Challenge
- Another one which is a bit non-generic if I may: AVEC - Audio/Visual Emotion Challenge

My thinking didnt address how well does that dataset represent a real world or a practical usecase scenario and what would I achieve by getting that x% accuracy in that dataset? So, in order to determine that I needed to do a proper analysis of the dataset with these thoughts in mind! But I hadn't inititally. Also at that time I hadn't had the chance to look at the dataset as such (coz it was invite only) but I kinda knew what exactly it had. 

From the interactions with my guides, I involved in qualitative analysis and discussions, thinking about various ways emotion is portrayed as well as expressed (by people/actors) in videos.  I came to realize that the problem statement when rightly put, is much more complex. And simplifying it to a trivial way like EmotiW wasnt really gonna cut it. Most of the explanation are based on some sort of entertaintment forms of video? Like movies or TV shows or Newscast etc rather than natural human interactions as such. I guess relating to the former was more easier when I was thinking and analyzing...

#### So, why does MEC or EmotiW not useful?

- EmotiW contains ~773 short videos. Maximum being 16 frames (which turned out to be only few milliseconds long video).
- A more qualitative study finds that most of the videos are 1-2 secs long (as seen from the media player) and just represents the facial expression. 
- A short quantitative anaylsis on the trainset show that its even worse, maximum being 640ms and minimum being 8ms while average is around 577ms

~~~python
In [16]: vidreader.get_meta_data()
Out[16]: 
{'duration': 0.6,
 'ffmpeg_version': u'3.3.1 built with gcc 4.8 (Ubuntu 4.8.4-2ubuntu1~14.04.3)',
 'fps': 25.0,
 'nframes': 15,
 'plugin': 'ffmpeg',
 'size': (720, 576),
 'source_size': (720, 576)}

In [17]: vidreader.get_meta_data()['fps']
Out[17]: 25.0

In [18]: length = []

In [19]: for f in files:
    ...:     vidreader = imageio.get_reader(f, 'ffmpeg')
    ...:     length.append(vidreader.get_meta_data()['duration'])
    ...:     

In [20]: len(length)
Out[20]: 773

In [21]: max(length)
Out[21]: 0.64

In [22]: min(length)
Out[22]: 0.08

In [23]: sum(length) / len(length)
Out[23]: 0.5779560155239367

~~~



- With the above details at hand one can say that the audio modality doesn't have much importance to offer in this multimodal dataset, the same is observed while qualitative analysis
- One must note that these are excellent datasets and credit goes to the organizers for making this available and useful to the community. We have indeed seen some progress in this sub-topic more solely because of these datasets. But probably the time has come where it isnt truly useful in the type of better multimodal research work which should be conducted now.
- More on the why these datasets aren't useful is mentioned after the discussion below.

To start off, we'll set our working conditions not very different from the datasets and try to analyze what improvements can be done. More on what current models are and what they achieve, given after below discussions.

- What is emotion detection in videos?
  - Take a video clip few seconds long (5 - 30 seconds)
    - Short clips (5-10 secs) 
    - Longer clips (20+ secs) 
- Thinking on the elements which are needed:
  - Simple - person, face, gesture, audio - laughing, awww etc (the characteristic sounds)
  - Complex - context, perception, perspectives eg: sarcasm, multiple ppl-emotions, comedy show

### Why is it complex?

Analyzing emotion has multiple aspects. Some of them which I've realized are given below. (If the language seems incorrect or vague, apologies, please give any suggestions or comments below without any hesitation. I'm also aware that there are lots of better research works which talks about these things in a more structured and correct way, I'm trying to find, read and explore them, if you have any suggestions please comment below. It'd help in better formulation and understanding too)
- Context: 
  - Is it a character's emotions [multiple characters - different emotions] or Scene's emotion as a whole or is it the context of viewer's perception. What do we call as emotion in each case.
  - Scene implies one emotion while activity implies another emotion. This is just 
  - In few ways tightly bound with perception.
  - Making the context clear or generating results for each context will make it more effective

- Perception 

  - **Within the video**: Character's emotion doesn't always imply the scene's emotion.
  - Tackling this isn't going to be easy as I believe there are loads of work going on in cognition and common sense understanding. We can atleast start with just describing the character's emotions first and then make use of other attributes useful for the scene's emotion.
  - There is also the prime example of **satire** and **standup** comedy shows, where the content told by the comedian as such means something else entirely than what it's perceived by the audience. The emotion of the video changes dramatically given the context that it's a satire.
  - We can formulate the above using a conditional variable which can be the context obtained from the attributes of the video, contents of the video etc.

- Perspectives

  - **Outside the video**: Often viewer's perception is scene's emotion as opposed to what the creator is trying convey through the video. But we can differentitate these two easily (might not be trivial always but).
  - For the above, both perceptions (this and the above one `within the video`) can be represented seperately with formulation from appropriate features. To make it more clear, let me give an example, an action fight scene which involves a character killing people (eg: with a sword) but its potrayed with a slow motion and a background music which reflects the sadness in his loss (due to which he takes a revenge). If portayed correctly, probably many people would get the emotion is supposed to be sadness if i may, but there would be a lot of people who would view it as anger, action, fight etc. In this case, audio plays such a huge role even if the facial expressions are of different emotion.
  - It can also be on how different people may interpret the same thing. Apart from the obvious examples, it also includes some things like - many scenes dont necessarily convey "an emotion". They show a scenario maybe? which can be interpreted in many ways.
  - Partly I can think of using the probabilities to explain a certain result. ex: if our softmax output has let's say 2 classes whose probabilties are equivalent then you can analyze it top down to see if the model is right in saying that those classes are interpreted equally etc. 

Analyzing these points is important to not only understanding but also in formulating the problem, which directly affects what sorts of dataset we create. Which also has also a direct impact on what research outputs we get as a community. Hopfeully some of the points I made for tackling the challenges makes sense, please comment your thoughts below. :). Changing the course now, towards current models.

### What does current models achieve?

- They train on short video clips from datasets mentioned above. They use CNNs for image level feature extraction and use RNNs for making use of temporal component present in videos. Some use a 3D convolution approaches for the same. Few people have combined both into a form of ensemble model by concatenating the features before passing it to the classifier. In fact, my initial approach was based on a paper which followed the above method.

- Audio modality
  -  They take the audio features directly from a library, based on a standard configuration for emotion recognition task. This is really just feature engineering from one pov.
  -  They use the audio features along with image-frame features (concat) and classify the combined features into one of the classes. Weightages are given and sometimes can also made learnable parameters. 

### What does it not address?

- It doesn't implement a truly multimodal approach.
  - My focus was to effectively use the multiple modalities, at the very least using the available modalities in the best way possible.
  - CNNs have certainly made possible the great understanding of images and continous frames to an extent, so using that is totally cool. But its time that we come up with better feature extraction methods for audio rather than using standard feature engineering methods. I partially beleive those methods exist already in research done in different sub-topics, most notably speech recognition and music information extraction to an extent.
  - Combination of modalities can and should be much more better, in terms of getting sophisticated results by exploiting simple concepts. Ex: How people are solving relational tasks in computer vision. [Raposo et al](https://arxiv.org/abs/1702.05068). I can imagine this being applied to specific things on interest from combination of video and audio modality (concurrence of an event, wrt complimenting each other)
  - Another example of a simple concept being exploited for unsupervised multimodal learning - [Arandjelović and Zisserman](https://arxiv.org/abs/1705.08168)

- Supervised training on EmotiW dataset

    - As explained above, the dataset is mainly facial expression focused, it doesn't have anything more to offer towards the different aspects mentioned above.
    - The audio modality as told before has no value as such, one might be able to find few clips where there is a small but significant sound for analysis, even then it doesn't help much as seen in the accuracies of image only, audio only and image+audio models [Vielzeuf et al](https://arxiv.org/abs/1709.07200). This can be seen as partly due to lack of better audio models but one can see while going through literature survey in this topic, that irrespective of the method, audio has little to contibute towards the improvement of accuracy.
    - This would cause the network to learn only face expressions, which can be achieved considerably well by image datasets itself with single modality.
    - Obviously, it fails to address any amount of perception problems, such as higher level understanding of scene's emotion or multiple character's emotions (this can actually be done using a trivial approach).

### Conclusion - "Dude, what are you trying to say?!"

I wish and propose that people should start working on addressing the truly multimodal aspects of understanding emotions in videos. The above gives a detailed analysis of why and little bit of how. I have refined my ideas and learned more through time and will share that in another post soon as I close down on deliverable points. It will bring in prespectives from Cognitive and Neuroscience as well. It will also focus on specific experimental settings for doing the better multimodal research on videos for emotion classification. 

I had continued the work (which started with the aim of creating a tool for multimodal emotion detection) in the direction of general multimodal understanding of videos with lot of focus on audio. It was done on simple examples using machine learning as well as deep learning methods. It is meant to be a starting point from which we can adapt to different problem areas and bigger datasets as well. It was also partly due to the thinking that those methods and research will directly be helpful for emotion understanding as well. I haven't yet had a chance to come back to focus on emotion understanding until recently.

##### Acknowledgments

Lot of discussions and understanding was gained by the guidance of [Jakob Suchan](http://cosy.informatik.uni-bremen.de/staff/jakob-suchan) and [Prof. Mehul Bhatt](http://www.mehulbhatt.org/), both at the University of Bremen, Germany. Thanks a lot to them.
