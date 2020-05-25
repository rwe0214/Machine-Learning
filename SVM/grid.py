import libsvm.python.svmutil as svmutil

def range_f(begin,end,step):
    # like range, but works on non-integer too
    seq = []
    while True:
        if step > 0 and begin > end: break
        if step < 0 and begin < end: break
        seq.append(begin)
        begin = begin + step
    return seq

def find_parameters(problem, options):
    if type(options) == str:
        options = options.split()
    i = 0

    fold = 5
    kernel_type = 2
    c_begin, c_end, c_step = -5,  15,  2
    g_begin, g_end, g_step =  3, -15, -2
    r_begin, r_end, r_step = -2,  15,  2
    d_end = 3
    grid_with_c, grid_with_g, grid_with_r, grid_with_d = True, False, False, False

    while i < len(options):
        if options[i] == '-t':
            i = i + 1
            kernel_type = options[i] 
        elif options[i] == '-log2c':
            i = i + 1
            if options[i] == 'null':
                grid_with_c = False
            else:
                c_begin, c_end, c_step = map(float,options[i].split(','))
        elif options[i] == '-log2g':
            grid_with_g = True
            i = i + 1
            if options[i] == 'null':
                grid_with_g = False
            else:
                g_begin, g_end, g_step = map(float,options[i].split(','))
        elif options[i] == '-d':
            grid_with_d = True
            i = i + 1
            if options[i] == 'null':
                grid_with_g = False
            else:
                d_end = int(options[i])
        elif options[i] == '-log2r':
            grid_with_r = True
            i = i + 1
            if options[i] == 'null':
                grid_with_r = False
            else:
                r_begin, r_end, r_step = map(float,options[i].split(','))
        i = i + 1
    
    c_seq = range_f(c_begin,c_end,c_step)
    g_seq = range_f(g_begin,g_end,g_step)
    r_seq = range_f(r_begin,r_end,r_step)
    d_seq = range_f(1,d_end,1)

    tables = []
    col = []

    if not grid_with_c:
        c_seq = [None]
    if not grid_with_g:
        g_seq = [None]
    if not grid_with_r:
        r_seq = [None]
    if not grid_with_d:
        d_seq = [None]
    

    job = []
    for i in range(len(c_seq)):
        for j in range(len(g_seq)):
            for k in range(len(r_seq)):
                for l in range(len(d_seq)):
                    job.append((c_seq[i], g_seq[j], r_seq[k], d_seq[l]))
    
    best_rate = 0
    best_c = 0
    bext_g = 0
    best_r = 0
    best_d = 0

    for i in range(len(job)):
        table = []
        cmd_line = '-q -t {0} -v {1}'.format(kernel_type, fold)
        if grid_with_c:
            table.append(2**job[i][0])
            cmd_line += ' -c {0}'.format(2**job[i][0])
        if grid_with_g:
            table.append(2**job[i][1])
            cmd_line += ' -g {0}'.format(2**job[i][1])
        if grid_with_r:
            table.append(2**job[i][2])
            cmd_line += ' -r {0}'.format(2**job[i][2])
        if grid_with_d:
            table.append(job[i][3])
            cmd_line += ' -d {0}'.format(job[i][3])
        table.append(cmd_line)
        print('[{:5d}/{:5d}]'.format(i+1, len(job)), end=' ')
        rate = svmutil.svm_train(problem, cmd_line)
        table.append(rate)
        tables.append(table)
        if best_rate < rate:
            best_rate = rate
            best_c = 2**int(job[i][0])
            if grid_with_g:
                best_g = 2**int(job[i][1])
            if grid_with_r:
                best_r = 2**int(job[i][2])
            if grid_with_d:
                best_d = job[i][3]
    ret = []
    ret.append(best_c)
    col.append('c')
    if grid_with_g:
        col.append('g')
        ret.append(best_g)
    if grid_with_r:
        col.append('r')
        ret.append(best_r)
    if grid_with_d:
        col.append('d')
        ret.append(best_d)
    ret.append(best_rate)
    col.append('options')
    col.append('rate')
    return ret, tables, col
