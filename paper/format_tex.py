import os
classes = [
    'DeGrooteFregly2016Muscle',
    'MocoStudy',
    'MocoStudies',
    'MocoProblem',
    'MocoSolver',
    'PositionMotion',
    'MocoSolution',
    'MocoTrajectory',
    'MocoControlGoal',
    'MocoControlBoundConstraint',
    'Model',
    'MocoTrajectories',
    'MocoInverse',
    'MocoTrack',
    'SmoothSphereHalfSpaceForce',
    'Coordinate',
    'MocoTropterSolver',
    'MocoCasADiSolver',
    'Storage',
]

# Manually correct "Simbody's Motion class"

with open('MocoPaper.tex', 'r') as old:
    with open('MocoPaper_formatted.tex', 'w') as new:
        for line in old:
            # if 'includegraphics' in line:
            #     new.write(this_line)
            #     continue
            this_line = line
            for c in classes:
                this_line = this_line.replace(c, '\\textit{%s}' % c)
            new.write(this_line)
os.system('mv MocoPaper_formatted.tex MocoPaper.tex')

