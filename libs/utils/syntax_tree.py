import Levenshtein

__all__ = ['SyntaxTree', 'cal_teds']

def cal_wer(pred_str, label_str):
    if len(label_str):
        return Levenshtein.distance(pred_str, label_str) / len(label_str)
    else:
        return 1e6

class Node():
    def __init__(self, tid, key, value='', latex_i=None, tree=None):
        self.tid = tid
        self.key = key
        self.value = value
        self.latex_i = latex_i
        self.tree = tree
        self.parent = None
        self.depth = 0 if tid == 'ROOT_TID' else len(tid) - len(tid.replace('.', '')) + 1
        self.children = []
        self.name = key

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    def set_value(self, value):
        self.value = value

    def __repr__(self):
        return str((self.tid, self.key, self.value))

class SyntaxTree():

    hierarchy_lst_sorted = ['section', 'subsection', 'subsubsection']

    def __init__(self, hierarchy_str, tree_name='TreeName'):
        self.tree_name = tree_name
        self.root = Node('ROOT_TID', 'root', latex_i=-1)
        self.nodes = [self.root]
        self.depth2hierarchy, self.hierarchy2depth, self.depth = self.init_hierarachy(hierarchy_str)
        self.has_acknowledge = False
        self.has_reference = False

    def init_hierarachy(self, hierarchy_str):
        hierarchy_str = list(set(hierarchy_str))
        depth = len(hierarchy_str)
        sort_hierarchy = sorted(hierarchy_str, key=lambda x:self.hierarchy_lst_sorted.index(x))
        hierarchy2depth = dict(zip(sort_hierarchy, range(1, len(sort_hierarchy) + 1)))
        depth2hierarchy = dict(zip(range(1, len(sort_hierarchy) + 1), sort_hierarchy))
        return depth2hierarchy, hierarchy2depth, depth

    def add_node(self, hierarchy_str=None, latex_i=None, key='', value='', depth_set=None):
        assert hierarchy_str or depth_set
        if hierarchy_str and cal_wer(key.replace(' ', '').lower(), 'acknowledgments'.lower()) < 0.3:
            hierarchy_str = self.depth2hierarchy[1]
        if not depth_set:
            depth = self.hierarchy2depth[hierarchy_str]
        else:
            depth = depth_set
        parent = None
        for n in self.nodes[::-1]:
            if depth > n.depth:
                parent = n
                break
        if parent.tid != 'ROOT_TID':
            tid = parent.tid + '.' + str(len(parent.children) + 1)
        else:
            tid = str(len(parent.children) + 1)
        
        node = Node(tid, key, value, latex_i)
        parent.add_child(node)
        self.nodes.append(node)
        match_wer = cal_wer(key.replace(' ', '').lower(), 'acknowledgments'.lower())
        if match_wer < 0.3:
            self.has_acknowledge = True
        match_wer = cal_wer(key.replace(' ', '').lower(), 'references'.lower())
        if match_wer < 0.3:
            self.has_reference = True

    def __repr__(self) -> str:
        output = '***{}***\n'.format(self.tree_name)
        blank = '\t'
        for node in self.nodes:
            prefix = blank * node.depth
            output += prefix + repr(node) + '\n'
        return output

    def __iter__(self):
        yield from self.nodes

    def __getitem__(self, idx):
        return self.nodes[idx]

    def __len__(self):
        return len(self.nodes)

    @classmethod
    def read_log(clc, lines):
        tree_name = list(filter(lambda x:len(x) - len(x.replace('***', '')) == 6, lines))[0].replace('***', '').replace('\n', '')
        lines_tree = list(filter(lambda x:'(' in x and ')' in x and ', ' in x, lines))
        depth = max([len(eval(l)[0]) - len(eval(l)[0].replace('.', '')) + 1 for l in lines_tree])
        tree = clc(clc.hierarchy_lst_sorted[:depth], tree_name)
        for l in lines_tree[1:]:
            tid, key, value = eval(l)
            tree.add_node(key=key, depth_set=len(tid) - len(tid.replace('.', '')) + 1, value=value)
        return tree

    @classmethod
    def read_hier(clc, hier_info_lst, pdf_path):
        depth = max([len(t['tid']) - len(t['tid'].replace('.', '')) + 1 for t in hier_info_lst])
        tree = clc(clc.hierarchy_lst_sorted[:depth], pdf_path)
        for l in hier_info_lst:
            tid, key = l['tid'], l['key']
            tree.add_node(key=key, depth_set=len(tid) - len(tid.replace('.', '')) + 1)
        return tree


if __name__ == "__main__":
    str_lst = ['section', 'subsection', 'subsubsection']
    tree = SyntaxTree(str_lst)
    str_input = ['section', 'subsection', 'section', 'subsection', 'subsubsection', 'subsubsection',
                 'subsection', 'section', 'subsection']
    for hierarchy in str_input:
        tree.add_node(hierarchy)
    print(tree)
