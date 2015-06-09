#What is it not possible to do by design?

##Bundle fly scan data into a step scan event
In the ``step_fly`` example above, it would not be possible to bundle the 
data from `fly_mtr` into a filestore entry and `fly_det` into a filestore 
entry and then insert those into the same event as `step_mtr`. While it not 
necessarily illogical to create such an event, we would argue that we already
provide tools to do this reconstruction at data analysis time.  Therefore, it
is not necessary to store the data in such a manner. Also it would add 
unnecessary complication to the bluesky RunEngine to do this.

What a flyscan+stepscan event would look like
```
Event
=====
data            :
  step_mtr        : [5.123029068072772, 1433188852.2006402] 
  fly_mtr         : 8ae46bff-b577-414a-b3a3-009548250d3c
  fly_det         : 9a715a80-007e-4591-b243-2535d66e3ee5
seq_num         : 5                                       
time            : 1433188852.2006402                      
time_as_datetime: 2015-06-01 16:00:52.200640              
uid             : 4ae29c60-0699-4492-82ec-ccc137811a68    
```
