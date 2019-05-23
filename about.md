---
layout: page
title: About
permalink: /
---


I am currently a Visiting Research Scholar at the [Machine Learning and Perception lab at Georgia Tech](http://mlp.cc.gatech.edu/) led by [Prof. Dhruv Batra](https://www.cc.gatech.edu/~dbatra/). I work on Machine Learning problems at the intersection of vision and language. Prior to that I was on a research internship at the [Statistics and Machine Learning Group at Indian Institute of Science, Bangalore](http://sml.csa.iisc.ernet.in/SML/). Before that I was working as a Platform Engineer at [Soroco](http://soroco.com/). I completed my Bachelor's in Computer Science and Engineering from the [Indian Institute of Technology Kharagpur](http://www.iitkgp.ac.in/) in May 2016.

<br>

## Selected Projects
<br>

<div class="container">
    

        <div class="row">
            <div class="col-md-12 col-sm-12 col-lg-12">
            <div class="row">
            
                <div class="col-md-12 col-sm-12 col-lg-12">
                    <b>Visual Landmark Selection for Generating Grounded and
                Interpretable Navigation Instructions.</b>
                </div>
                
            </div>
                
            <div class="row">
                
                <div class="col-md-12 col-sm-12 col-lg-12">
                
                   <b> S. Agarwal </b>, 
                    <a href="https://www.cc.gatech.edu/~parikh/" target="_blank">D. Parikh</a>,
                    <a href="https://www.cc.gatech.edu/~dbatra/" target="_blank">D. Batra</a>,
                    <a href="https://panderson.me/" target="_blank">P. Anderson</a>,
                    <a href="https://www.cc.gatech.edu/~slee3191/" target="_blank">S. Lee</a>
                    
                    <br>
    
                    CVPR 2019 Workshop on Deep Learning for Semantic Visual Navigation
                    <br>
                    <a href="/files/cvpr-workshop.pdf" target="_blank">pdf</a>
                    /
                    <a >code(to be released)</a>          
                </div>
            </div>
            </div>
            
    </div>

    <div class="row">
        <div class="col-md-4 col-sm-4 col-lg-4">
            <div><img align="left" src="/images/projects/cvpr_model.png" width="100%"></div>
            <div style="text-align: center;">Two-stage instruction generation</div>
        </div>
        <div class="col-md-8 col-sm-8 col-lg-8">
            <div style="font-size:14px">
                    Instruction following for vision-and-language navigation
                    (VLN) has prompted significant research efforts developing
                    more powerful “follower” models since its inception.
                    However, the inverse task of generating visually grounded
                    instructions given a trajectory – or learning a “speaker”
                    model – has been largely unexamined. We present a “speaker”
                    model that generates navigation instructions in two stages,
                    by first selecting a series of discrete visual landmarks
                    along a trajectory using hard attention, and then second
                    generating language instructions conditioned on these
                    landmarks. This two-stage approach improves over prior work,
                    while also permitting greater interpretability. We hope to
                    extend this to a reinforcement learning setting where
                    landmark selection is optimized to maximize a follower’s
                    performance without disrupting the model’s language fluency.
                
            </div>
        </div>
    </div>
    
    
    
    <br>
    <br>
    
                
        <div class="row">
            <div class="col-md-12 col-sm-12 col-lg-12">
            <div class="row">
            
                <div class="col-md-12 col-sm-12 col-lg-12">
                    <b>Fast GPU-Based Simulator for Room-to-Room dataset</b>
                </div>
                
            </div>
                
            <div class="row">
                
                <div class="col-md-12 col-sm-12 col-lg-12">
                    <a >code(to be released)</a>
                </div>
            </div>
            </div>
            
    </div>
        
        
    

    <div class="row">
        <div class="col-md-4 col-sm-4 col-lg-4">
            <div><img align="left" src="/images/projects/r2r.gif" width="100%"></div>
            <div style="text-align: center;"> Sample trajectory in Room-to-Room dataset</div>
        </div>
        <div class="col-md-8 col-sm-8 col-lg-8">
            <div style="font-size:14px">
                Room-to-Room <a href="https://bringmeaspoon.org/" target="_blank"> dataset </a> is a commond dataset used in several vision and language navigation tasks. The dataset contains real-world panoramic scans of building interiors provided my the <a href="https://niessner.github.io/Matterport/" target="_blank"> Matterport3D dataset </a>. An agent can choose to move in this 3D environment by taking actions. I optimized the original <a href="https://github.com/peteanderson80/Matterport3DSimulator" target="_blank"> Room-to-Room simulator </a>
                to use GPU for state update operations when the agent takes any actions. Combined with caching of repeated computations this resulted in a simulator that is <b> 17x </b> faster frame rate on a single GPU! Training time of most vision-and-language models can be brought down significantly using this simulator instead of the original simulator.
                
            </div>
        </div>
    </div>
    



<br>
<br>




    
    
                
        <div class="row">
            <div class="col-md-12 col-sm-12 col-lg-12">
            <div class="row">
            
                <div class="col-md-12 col-sm-12 col-lg-12">
                    <b>AppTechMiner: Mining Applications and Techniques from Scientific Articles</b>
                </div>
                
            </div>
                
            <div class="row">
                
                <div class="col-md-12 col-sm-12 col-lg-12">
                
                    M. Singh,
                    S. Dan,
                    <b> S. Agarwal </b>, 
                    P. Goyal,
                    A. Mukherjee,
                    
                    <br>
    
                    Joint Conference on Digital Libraries (JCDL) 2017:6th International Workshop On Mining Scientific Publications
                    <br>
                    <a href="https://arxiv.org/abs/1709.03064" target="_blank">pdf</a>                   
                </div>
            </div>
            </div>
            
    </div>

    <div class="row">
        <div class="col-md-4 col-sm-4 col-lg-4">
            <div><img align="left" src="/images/projects/apptechminer.png" width="100%"></div>
            <div style="text-align: center;"> A sample Phrase-Cloud representing the proportion of papers for an area.</div>
        </div>
        <div class="col-md-8 col-sm-8 col-lg-8">
            <div style="font-size:14px">
                A rule-based information extraction framework that automatically constructs a knowledge base of all application areas and problem solving "techniques", given a text corpus of research papers. "Techniques" include tools, methods, datasets or evaluation metrics. We also categorize individual research articles based on their application areas and the techniques proposed/improved in the article. Our system achieves high average precision (~82%) and recall (~84%) in knowledge base creation. It also performs well in application and technique assignment to an individual article (average accuracy ~66%). We demonstrated the framework for the domain of computational linguistics but this can be easily generalized to any other field of research.
                
            </div>
        </div>
    </div>
    
    
    
</div>

