library(DiagrammeR)

grViz("
digraph G {

graph [
  layout = dot,
  rankdir = LR,
  fontsize = 24,
  labelloc = t,
  label = 'Learning objectives recover distinct latent structures'
]

node [
  shape = box,
  style = rounded,
  fontsize = 18,
  fontname = Helvetica,
  color = gray30,
  penwidth = 1.5,
  width = 2.2,
  height = 0.7
]

edge [
  arrowsize = 0.8,
  penwidth = 1.5,
  color = gray40
]

 
# Left branch
 

same1 [label='Same\npsychometric\ndata']

obj1 [label='Behavior-oriented\nobjective']

latent1 [label='Latent\nSpace A']

out1 [label='Behavioral\nPhenotypes',
      shape=ellipse,
      style='filled',
      fillcolor='lightgoldenrod1']

 
# Right branch
 

same2 [label='Same\npsychometric\ndata']

obj2 [label='Alignment-oriented\nobjective']

latent2 [label='Latent\nSpace B']

out2 [label='Teacher–Child\nCorrespondence',
      shape=ellipse,
      style='filled',
      fillcolor='lightcyan']

 
# Connections
 

same1 -> obj1
obj1 -> latent1
latent1 -> out1

same2 -> obj2
obj2 -> latent2
latent2 -> out2

 
# Bottom statement
 

conclusion [
label='Representation learning objective shapes the recovered latent structure',
shape=box,
style='rounded,filled',
fillcolor='gray95',
fontsize=20
]

out1 -> conclusion [arrowhead=none]
out2 -> conclusion [arrowhead=none]

{rank=same; same1 same2}
{rank=same; obj1 obj2}
{rank=same; latent1 latent2}
{rank=same; out1 out2}

}
")