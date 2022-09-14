import glob, os

file_list = [f.split('/')[-1] for f in glob.glob("config_NRPMw*.ini")]

rename_ini = 0
insert     = 0
replace    = 1
remove     = 0

if(rename_ini):
    
    wrong_string   = '0M.ini'
    correct_string = '0M_SEED_0000.ini'
    
    print("String to be changed:", wrong_string)
    for f in file_list:
        new_name = f.replace(wrong_string, correct_string)
        print("Old name:", f)
        print("New name:", new_name)
        os.rename(f, new_name)

if(insert):
    
    text_previous_line = 'tides       = 1'
    text_to_insert     = 'm-q-par     = 1\nmc-q-par    = 0'
    
    index_prev_line = 0 
    for target_file in file_list:
        with open(target_file, 'r+') as fd:
            contents = fd.readlines()
            for i,line in enumerate(contents):
                if(text_previous_line in line):
                    index_prev_line = i+1
            assert not(index_prev_line==0), "Could not find text after which the substitution is suppposed to take place."
        fd.close()
        contents.insert(index_prev_line, text_to_insert+'\n')
        f = open(target_file, 'w')
        contents = ''.join(contents)
        f.write(contents)
        f.close()

if(replace):
    
    wrong_string   = 'q-max           = 1.1'
    correct_string = 'q-max           = 2.0'
    
    for target_file in file_list:
        with open(target_file, 'r') as file:
            filedata_event_file = file.read()
            filedata_event_file = filedata_event_file.replace(wrong_string, correct_string)
        with open(target_file, 'w') as file:
            file.write(filedata_event_file)

if(remove):

    wrong_string = 'mc-q-par'
    for target_file in file_list:
        with open(target_file, "r") as f:
            lines = f.readlines()
        with open(target_file, "w") as f:
            for line in lines:
                if not (wrong_string in line.strip("\n")):
                    f.write(line)
