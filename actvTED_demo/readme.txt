++: runnable scripts
--: matlab functions
^^: data in MAT file

=============================================

* Demo on Toy Data
  ++ demo_dataselection_4gaussmix.m   // demo program on 4 gaussians in a 2-D space
  ++ demo_dataselection_manudraw.m    // demo program on manually drawn data
  -- fourgaussian.m                   // generator of four gaussians
  -- drawdata.m                       // program to manually draw data

* Simulation on Newsgroup Text Classification:
  ++ simu_newsgroup.m                 // the experiment on newsgroup data
  -- transdesign_kernelridge.m        // transductive experimental design simulation
  -- randomsamp_kernelridge.m         // random sampling simulation
  ^^ newsgroup_4c_normalized.mat      // the used data with tf-idf features

* Transductive Experimental Design
  -- transdesign_sq.m                 // sequential transductive exerimental design

* Other Supporting Functions
  ... 
