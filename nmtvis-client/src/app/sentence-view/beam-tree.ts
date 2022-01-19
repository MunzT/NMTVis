import * as d3 from 'd3';
import {Constants} from '../constants';
import {BeamNode} from './beam-node';
import {DocumentService} from '../services/document.service';

export class BeamTree {

    treemap;
    root;
    svg;
    colorScale;
    colors;
    zoom;
    zoomLevel = 1;
    colorLegend;
    hoveredNode;
    focusNode;
    currentInput = "";
    currentFocusChildIndex = 0;
    height;

    constructor(private treeData: any, private that: any, readonly documentService: DocumentService) {

    }

    build() {
        var margin = {top: 50, right: 30, bottom: 50, left: 50},
            width = 1650 - margin.left - margin.right,
            height = 500 - margin.top - margin.bottom;
        this.height = height;

        // append the svg object to the body of the page
        // appends a 'group' element to 'svg'
        // moves the 'group' element to the top left margin
        d3.select("#tree-vis").selectAll('*').remove();
        this.svg = d3.select("#tree-vis")
            .attr("width", width + margin.right + margin.left)
            .attr("height", height + margin.top + margin.bottom)
            .style("background", "#fff")
            .style("border", "2px solid #ddd")
            .style("border-radius", "5px")
            .style("margin-bottom", "15px")
            .attr("id", "tree-vis")
            .append("g");

        var tree = this;

        this.zoom = d3.zoom().scaleExtent([0.5, 2]).on("zoom", function () {
            tree.zoomLevel = d3.event.transform.k;
            tree.svg.attr("transform", d3.event.transform);
        });

        d3.select('#tree-vis')
            .on("mousedown", function () {
            })
            .call(this.zoom)
            .on("dblclick.zoom", null)
            .call(this.zoom.translateBy, margin.left, margin.top)

        d3.select('#zoom-in').on('click', function () {
            // Smooth zooming
            tree.zoom.scaleBy(d3.select('#tree-vis').transition().duration(500), 1.3);
        });

        d3.select('#zoom-out').on('click', function () {
            // Ordinal zooming
            tree.zoom.scaleBy(d3.select('#tree-vis').transition().duration(500), 1 / 1.3);
        });

        d3.select('body').on("keydown", function () {
            tree.keydown(tree);
        });

        var i = 0,
            duration = 500,
            root;

        // declares a tree layout and assigns the size
        this.treemap = d3.tree<BeamNode>().size([height, width]).separation(function (a, b) {
            return a.parent == b.parent ? 2 : 3;
        });

        var colorOption = 1;
        var domain = [];
        if (colorOption == 0) {
            this.colors = ['#FCE4EC', '#F8BBD0', '#EB3B5A'];
            domain = [-3, -1.5, 0];
        }
        else if (colorOption == 1) {
            this.colors = ['#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f'];
            domain = [-3, -2, -0.5, -0.05, -0.005, -0.0005];
        }
        else if (colorOption == 2) {
            this.colors = ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6'].reverse();
            for (var i = -2.0; i <= 0; i += 3.0 / (this.colors.length - 1)) {
                domain.push(i);
            }
        }
        else if (colorOption == 5) {
            this.colors = ['#fee3d7', '#fca98d', '#dd2924', '#68000d'];
            domain = [-3, -0.5, -0.05, -0.0005];
        }
        else if (colorOption == 6) {
            this.colors = ['#fee3d7', '#fdcfbc', '#fc9677', '#d3201f', '#a70f15', '#69000c'];
            domain = [-3, -2, -0.5, -0.05, -0.005, -0.0005];
        }

        this.colorScale = d3.scaleLinear().domain(domain).range(this.colors).clamp(true);
        this.buildColorLegend();

        this.treeData.is_golden = true;

        // Assigns parent, children, height, depth
        this.root = d3.hierarchy(this.treeData, function (d) {
            return d.children;
        });
        this.root.x0 = height / 2;
        this.root.y0 = 0;

        this.update(this.root);
        this.center(this.root, "");
        var node = this.lastNodeOfGoldenHypothesis(this.root);
        this.click(node, this.getNodeSelection(node));
        this.mouseout(node, this.getNodeSelection(node));
    }

    updateData(treeData, highlightLastNode = false) {
        this.treeData = treeData;
        // Assigns parent, children, height, depth
        this.root = d3.hierarchy(this.treeData, function (d) {
            return d.children;
        });
        this.root.x0 = this.height / 2;
        this.root.y0 = 0;

        this.update(this.root);

        if (highlightLastNode) {
            var node = this.lastNodeOfGoldenHypothesis(this.root);
            this.click(node, this.getNodeSelection(node));
        }
    }

    calculateTextWidth(text) {
        var text = this.svg.append("text").text(text).style("font-size", "14px");
        var width = text.node().getComputedTextLength();
        text.remove();

        return width;
    }

    update(source) {

        // Assigns the x and y position for the nodes
        var treeData = this.treemap(this.root);
        var that = this.that;
        var beamTree = this;
        var duration = 600;

        // Compute the new tree layout.
        var nodes = treeData.descendants(),
            links = treeData.descendants().slice(1);

        var scale = d3.scaleLinear().domain([1, 10]).range([55, 0]);

        nodes.forEach(function (d) {
            if (!d.children || d.parent) {
                let multiChildOffset = d.parent.children.length > 1 ? -15 : 0;
                let textWidth = beamTree.calculateTextWidth(that.decodeText(d.data.name));
                let parentWidth = beamTree.calculateTextWidth(that.decodeText(d.parent.data.name));
                var padding = 15;

                if (d.parent.data.name.endsWith("@@") && !d.data.isCandidate) {
                    padding = 0;
                }
                d.y = d.parent.y + 0.5 * textWidth + 0.5 * parentWidth + padding - multiChildOffset;
            } else {
                d.y = 0;
            }
        });

        // ****************** Nodes section ***************************

        // Update the nodes...
        var node = this.svg.selectAll('g.node')
            .data(nodes, function (d: any) {
                return beamTree.getNodeId(d);
            });

        // Enter any new modes at the parent's previous position.
        var nodeEnter = node.enter().append('g')
            .attr('class', 'node')
            .attr("transform", function (d) {
                if (d.parent && d.parent.y !== undefined) {
                    return "translate(" + d.parent.y + "," + d.parent.x + ")";
                }
                return "translate(" + source.y0 + "," + source.x0 + ")";
            })
            .attr("node-path", function (d) {
                if (beamTree.getNodeId(beamTree.focusNode) === beamTree.getNodeId(d)) {
                    beamTree.focusNode = d;
                }
                return beamTree.getNodeId(d);
            })
            .on('click', function (d) {
                beamTree.click(d, this);
            })
            .on('mouseover', function (d) {
                beamTree.mouseover(d, this);
            })
            .on('mousedown', function (d) {
                // Stop panning when node is clicked, otherwise annoying the user
                d3.event.stopImmediatePropagation();
            })
            .on('mouseout', function (d) {
                beamTree.mouseout(d, this);
            });
        ;

        // Add Circle for the nodes
        nodeEnter.append('circle')
            .attr('class', function (d) {
                return "node";
            })
            .attr('r', 8)
            .style("fill", function (d) {
                return beamTree.colorScale(d.data.logprob);
            });

        // Add labels for the nodes
        nodeEnter.append('text')
            .attr("dy", "-.8em")
            .attr("x", function (d) {
                return -1;
            })
            .attr("text-anchor", function (d) {
                //return d.children ? "end" : "start";
                return "middle";
            })
            .attr("text-decoration", function (d) {
                return d.data.isEdit ? "underline" : "none";
            })
            .text(function (d) {
                var logprob = d.data.logprob ? d.data.logprob.toString() : "";

                var path = beamTree.getPath(d);
                if (path in that.correctionMap && that.correctionMap[path] === d.data.name) {
                    // This nodes was corrected, e.g. insert a '*'
                    return that.decodeText(d.data.name);
                }

                return that.decodeText(d.data.name);
            });

        // UPDATE
        var nodeUpdate = nodeEnter.merge(node);

        // Transition to the proper position for the node
        nodeUpdate.transition()
            .duration(duration)
            .attr("transform", function (d) {
                return "translate(" + d.y + "," + d.x + ")";
            })
            .attr("text-anchor", function (d) {
                //return d.children ? "end" : "start";
                return "middle";
            })
            .each(function (d) {
                if (beamTree.getNodeId(beamTree.focusNode) === beamTree.getNodeId(d)) {
                    beamTree.focusNode = d;
                }
            });

        nodeUpdate.select('text')
            .attr("x", function (d) {
                return -1;
            })
            .attr("text-anchor", function (d) {
                return "middle";
            })
            .text(function (d) {
                if (d.data.name === Constants.EOS) {
                    return "EOS";
                }
                if (d.data.name === Constants.SOS) {
                    return "SOS";
                }

                var logprob = d.data.logprob ? d.data.logprob.toString() : "";

                var path = beamTree.getPath(d);
                if (path in that.correctionMap && that.correctionMap[path] === d.data.name) {
                    // Could add '*' here
                    return that.decodeText(d.data.name);
                }

                return that.decodeText(d.data.name);
            });
        // Update the node attributes and style
        nodeUpdate.select('circle.node')
            .attr('r', function (d) {
                if (d.data.name.endsWith("@@") || (d.parent && d.parent.data.name.endsWith("@@"))) {
                    return 7;
                }
                return 8;
            })
            .style("fill", function (d) {
                return beamTree.colorScale(d.data.logprob);
            })
            .attr('cursor', 'pointer');

        // Remove any exiting nodes
        var nodeExit = node.exit().transition()
            .duration(duration)
            .attr("transform", function (d) {
                if (d.parent) {
                    return "translate(" + d.parent.y0 + "," + d.parent.x0 + ")";
                }
                return "translate(" + source.y + "," + source.x + ")";
            })
            .remove();

        // On exit reduce the node circles size to 0
        nodeExit.select('circle')
            .attr('r', 1e-6);

        // On exit reduce the opacity of text labels
        nodeExit.select('text')
            .style('fill-opacity', 1e-6);

        // ****************** links section ***************************

        // Update the links...
        var link = this.svg.selectAll('path.link')
            .data(links, function (d: any) {
                return beamTree.getNodeId(d);
            });

        // Enter any new links at the parent's previous position.
        var linkEnter = link.enter().insert('path', "g")
            .attr("class", function (d) {
                var classes = ["link"];
                if (d.data.isCandidate) {
                    classes.push("candidate-link")
                } else if (d.data.is_golden) {
                    classes.push("golden-link");
                }
                if (d.parent.data.name.endsWith("@@")) {
                    classes.push("bpe-link");
                }
                if (beamTree.isParentInFocus(d)) {
                    classes.push("focus-link");
                }
                return classes.join(" ");
            })
            .attr("link-path", function (d) {
                return beamTree.getNodeId(d);
            })
            .attr('d', function (d) {
                var o = {x: source.x0, y: source.y0};
                if (d.parent) {
                    o = {x: d.parent.x, y: d.parent.y};
                }
                return beamTree.diagonal(o, o)
            });

        // UPDATE
        var linkUpdate = linkEnter.merge(link);

        // Transition back to the parent element position
        linkUpdate
            .transition()
            .duration(duration)
            .attr("class", function (d) {
                var classes = ["link"];
                if (d.data.isCandidate) {
                    classes.push("candidate-link")
                } else if (d.data.is_golden) {
                    classes.push("golden-link");
                }
                if (d.parent.data.name.endsWith("@@")) {
                    classes.push("bpe-link");
                }
                if (beamTree.isParentInFocus(d)) {
                    classes.push("focus-link");
                }
                return classes.join(" ");
            })
            .attr('d', function (d) {
                return beamTree.diagonal(d, d.parent)
            });

        // Remove any exiting links
        var linkExit = link.exit()
            .transition()
            .duration(duration)
            .attr('d', function (d) {
                var o = {x: source.x, y: source.y};
                if (d.parent) {
                    o = {x: d.parent.x, y: d.parent.y};
                }
                return beamTree.diagonal(o, o)
            })
            .remove();

        // Store the old positions for transition.
        nodes.forEach(function (d: any) {
            d.x0 = d.x;
            d.y0 = d.y;
        });
    }

    isParentInFocus(d) {
        return d.parent && d.parent.children[this.currentFocusChildIndex] === d
            && this.getNodeId(this.focusNode) === this.getNodeId(d.parent);
    }

    addToDocumentUnkMap(d, word) {
        var that = this.that;
        var maxWordIndex = this.getMaxAttnSourceWordIndex(d);
        var maxAttn = d.data.attn[0][maxWordIndex];
        var maxWord = that.sentence[maxWordIndex];

        if (maxAttn > 0.9 && maxWordIndex < that.sentence.length) {
            if (!(maxWord in that.documentUnkMap)) {
                that.documentUnkMap[maxWord] = [];
            }
            if (that.documentUnkMap[maxWord].indexOf(word) === -1) {
                that.documentUnkMap[maxWord].push(word);
            }
        }
    }

    visitAncestors(d, fn) {
        if (!d) {
            return;
        }
        fn(d);
        this.visitAncestors(d.parent, fn);
    }

    // Toggle children on click.
    click(d, el) {
        var tree = this;
        this.that.addEvent("node-click", this.getPath(d) + " " + d.data.name);

        if (d.data.name === Constants.EOS && !d.data.isCandidate) {
            this.center(d, "");

            let eos = this.getNodeSelection(d);
            eos.select("circle")
                .transition().duration(300)
                .attr("r", 10)
                .transition().duration(400)
                .attr("r", 8);

            d3.selectAll('.focus-path').classed("focus-path", false);
            this.visitAncestors(d, node => {
                tree.getNodeSelection(node).classed("focus-path", true);
            });

            this.resetGoldenHypothesisBeam(this.that.beam);
            this.setGoldenHypothesis(d);
            this.setGoldenHypothesisBeam(this.that.beam, this.getPathList(d));
            this.that.attention = this.getBeamAttention(d);
            this.that.translation = this.getRawTranslation(d);
            this.that.updateTranslation(this.that.sentence.join(" "), this.that.translation.join(" "));
            this.that.updateAttentionMatrix(this.that.sentence.join(" "), this.that.translation.join(" "));
            return;
        }

        if (d.data.name === "EDIT") {
            if (this.focusNode === d) {
                this.center(d.parent, "left");
            }

            var that = this.that;
            that.partial = this.getPath(d);

            d = d.parent;
            if (d.data.attn && d.data.attn[0]) {
                that.beamAttention = [];
                that.beamAttention = d.data.attn[0].slice(0, that.sentence.length);
            }
            this.update(d);
            return;
        }

        if (this.isCandidateNode(d)) {
            if (this.focusNode === d) {
                this.center(d.parent, "left");
            }
        }

        if (this.isCandidateNode(d) && d.parent && d.parent.data.name === "UNK") {
            // UNK replacement
            this.that.unkMap[this.getPath(d.parent)] = d.data.name;

            this.addToDocumentUnkMap(d.parent, d.data.name);

            this.that.onCorrectionChange(d.data.name);
            this.that.addEvent("unk-replace", this.getPath(d) + "=>" + d.data.name);
        }
        else if (this.isCandidateNode(d)) {
            this.addToCorrectionMap(d.data.name, this.getPath(d));
            if (this.getPath(d) in this.that.unkMap) {
                delete this.that.unkMap[this.getPath(d)];
            }
            this.that.onCorrectionChange(d.data.name);
            this.that.addEvent("beam-replace", this.getPath(d) + "=>" + d.data.name);
        } else if (this.hasCandidateNodes(d)) {
            this.removeCandidateNodes(d, el);
        } else {
            this.addCandidateNodes(d, el);
        }
    }

    padPunctuation(s) {
        // Spaces around special characters and trim
        s = s.replace(/(!|\?|;|\.|,|;|:|"|'|\(|\))/g, " $1 ").trim()
        s = s.replace(/\s{2,}/g, " ")
        return s;
    }

    addToCorrectionMap(correction, partial) {
        this.that.correctionMap = {};
        this.that.correctionMap[partial] = this.that.encodeText(this.padPunctuation(correction));
    }

    getBeamNode(node, path) {
        if (path.length === 0) {
            return node;
        }
        for (var i = 0; i < node.children.length; i++) {
            if ((node.children[i].isEdit && path[0] === "<EDIT>") || node.children[i].name === path[0]) {
                return this.getBeamNode(node.children[i], path.slice(1));
            }
        }
    }

    isCandidateNode(d) {
        var beamNode = this.getBeamNode(this.treeData, this.getPathList(d).slice(1));
        return beamNode.isCandidate === true;
    }

    hasEditChild(d) {
        if (!d.children) {
            return false;
        }
        for (let child of d.children) {
            if (child.data.isEdit) {
                return true;
            }
        }
        return false;
    }

    getEditChild(d) {
        for (let child of d.children) {
            if (child.data.isEdit) {
                return child;
            }
        }
        return null;
    }

    removeEditChild(d) {
        if (!d) {
            return;
        }

        var index = -1;
        for (var i = 0; i < d.children.length; i++) {
            if (d.children[i].isEdit) {
                index = i;
            }
        }

        if (index > -1) {
            d.children.splice(index, 1);
        }
    }

    hasCandidateNodes(d) {
        var beamNode = this.getBeamNode(this.treeData, this.getPathList(d).slice(1));
        for (var i = 0; i < beamNode.children.length; i++) {
            if (beamNode.children[i].isCandidate) {
                return true;
            }
        }
        return false;
    }

    removeCandidateNodes(d, el) {
        var beamNode = this.getBeamNode(this.treeData, this.getPathList(d).slice(1));
        var children = [];

        for (var i = 0; i < beamNode.children.length; i++) {
            if (!beamNode.children[i].isCandidate) {
                children.push(beamNode.children[i]);
            }
        }
        beamNode.children = children;
        this.updateData(this.treeData);
    }

    getMaxAttnSourceWordIndex(d) {
        var maxSourceIndex = 0;
        for (var i = 1; i < d.data.attn[0].length; i++) {
            if (d.data.attn[0][i] > d.data.attn[0][maxSourceIndex]) {
                maxSourceIndex = i;
            }
        }
        return maxSourceIndex;
    }

    addCandidateNodesUNK(d, el) {
        let maxSourceIndex = this.getMaxAttnSourceWordIndex(d);
        var candidates = [];
        let attentionThreshold = 0.9;

        if (d.data.attn[0][maxSourceIndex] > attentionThreshold && maxSourceIndex < this.that.sentence.length) {
            var maxWord = this.that.sentence[maxSourceIndex];

            if (candidates.indexOf(maxWord) === -1) {
                candidates.push(maxWord);
            }

            if (maxWord in this.that.documentUnkMap) {
                let possibleTranslations = this.that.documentUnkMap[maxWord];
                for (var i = 0; i < possibleTranslations.length; i++) {
                    if (candidates.indexOf(possibleTranslations[i]) === -1) {
                        candidates.push(possibleTranslations[i]);
                    }
                }
            }
        }

        var beamNode = this.getBeamNode(this.treeData, this.getPathList(d).slice(1));
        for (var i = 0; i < candidates.length; i++) {
            beamNode.children.push({
                attn: [],
                name: candidates[i],
                children: [],
                logprob: -3,
                candidates: [],
                isCandidate: true
            });
        }
        this.updateData(this.treeData);
    }

    addCandidateNodes(d, el) {
        var element = d3.select(el);

        if (d.data.name === "UNK") {
            this.addCandidateNodesUNK(d, el);
            return;
        }

        var candidates = [];
        var existingChildren = [];
        for (var i = 0; i < d.data.children.length; i++) {
            existingChildren.push(d.data.children[i].name);
        }
        for (var i = 0; i < d.data.children.length; i++) {
            for (var j = 0; j < d.data.children[i].candidates.length; j++) {
                var candidate = d.data.children[i].candidates[j];
                if (existingChildren.indexOf(candidate) === -1 && candidates.indexOf(candidate) === -1) {
                    candidates.push(candidate);
                }
            }
        }

        var that = this.that;

        // 1. Get all candidates from children, excluding existing ones
        // 2. Push all candidates as children
        // 3. Update

        var beamNode = this.getBeamNode(this.treeData, this.getPathList(d).slice(1));
        for (var i = 0; i < candidates.length; i++) {
            beamNode.children.push({
                attn: [],
                name: candidates[i],
                children: [],
                logprob: -3,
                candidates: [],
                isCandidate: true
            });
        }
        this.updateData(this.treeData);
    }

    mouseover(d, el) {
        if (d.data.name === Constants.EOS && !d.data.isCandidate) {
            this.click(d, el);
        }
        this.onMouseover(d, el);
        this.center(d, "");
        d3.selectAll('.focus-path').classed("focus-path", false);
        this.visitAncestors(d, node => {
            this.getNodeSelection(node).classed("focus-path", true);
        });
    }

    onMouseover(d, el) {
        var that = this.that;
        that.clearAttention();
        that.addEvent("node-hover", this.getPath(d) + " " + d.data.name);

        if (!d.data.isCandidate && d.data.attn) {
            that.beamAttention = d.data.attn.slice(0, that.sentence.length);
            that.onCurrentAttentionChange();
            that.highlightTargetNode(d.depth);
        }

        if (this.hoveredNode) {
            that.highlightTargetNode(d.depth);
        }

        this.hoveredNode = el;
    }

    mouseout(d, el) {
        var element = d3.select(el);
        this.that.clearAttentionSelection();
        d3.selectAll(".fade-out").classed("fade-out", false);
        d3.selectAll('.attention-selected').classed("attention-selected", false);
    }

    center(d, dir) {
        // Unfocus old node
        this.getNodeSelection(this.focusNode).classed("focus", false);
        d3.select(".focus-link").classed("focus-link", false);
        // Unfocus path if not moving forwards
        if (dir === "left" || dir === "up" || dir === "down") {
            this.getNodeSelection(this.focusNode).classed("focus-path", false);
        }

        // Focus current node
        this.focusNode = d;
        var node = this.getNodeSelection(d);
        node.classed("focus", true);
        node.classed("focus-path", true);

        // Update attention view
        if (!this.focusNode.data.isCandidate && !this.focusNode.data.isEdit) {
            this.that.attention = this.getBeamAttention(this.focusNode);
            this.that.translation = this.getRawTranslation(this.focusNode);
            this.that.updateTranslation(this.that.sentence.join(" "), this.that.translation.join(" "));

            var tempNode = this.lastNodeOfGoldenHypothesis(this.focusNode);
            if (tempNode === this.focusNode) {
                tempNode = this.lastNodeOfBestHypothesis(this.focusNode);
            }
            var tempTranslation = this.getRawTranslation(tempNode);
            this.that.updateAttentionMatrix(this.that.sentence.join(" "), tempTranslation.join(" "));
        }
        // Center view on focus node
        var transform = this.getTransformation(this.svg.attr("transform"));
        var xTranslate = 850 - (transform.translateX + d.y);
        var yTranslate = 195 - (transform.translateY + d.x);
        if (dir && (transform.translateX + d.y > 1200 || transform.translateX + d.y < 50)) {
            this.zoom.translateBy(d3.select('#tree-vis').transition().duration(1000), xTranslate, 0);
        }
    }

    getNodeSelection(d) {
        return d3.select('[node-path="' + this.getNodeId(d) + '"]');
    }

    getLinkSelection(d) {
        return d3.select('[link-path="' + this.getNodeId(d) + '"]');
    }

    keydown(tree) {
        if (d3.event.getModifierState && d3.event.getModifierState('Control') && d3.event.keyCode == 13) {
            this.mouseover(this.lastNodeOfGoldenHypothesis(this.root), this.getNodeSelection(this.lastNodeOfGoldenHypothesis(this.root)));
            this.center(this.lastNodeOfGoldenHypothesis(this.root), "");
            this.that.onAcceptTranslation();
            return;
        }

        switch (d3.event.keyCode) {
            case 13: // Enter
                this.currentInput = "";
                this.click(this.focusNode, this.getNodeSelection(this.focusNode))
                break;
            case 37: { // Left Arrow
                if (this.focusNode.data.name === Constants.SOS) {
                    // jump from SOS to EOS
                    this.mouseover(this.lastNodeOfGoldenHypothesis(this.root), this.getNodeSelection(this.lastNodeOfGoldenHypothesis(this.root)));
                    this.center(this.lastNodeOfGoldenHypothesis(this.root), "");
                }
                else {
                    this.currentInput = "";
                    this.currentFocusChildIndex = 0;
                    tree.focusNode.parent ? this.center(tree.focusNode.parent, "left") : 0;
                }
                d3.event.view.event.preventDefault();
                break;
            }
            case 8: { // Backspace
                // BACKSPACE_KEY was fired in <input id="textbox">
                if (this.currentInput.length > 1) {
                    this.currentInput = this.currentInput.slice(0, -1);
                } else {
                    this.currentInput = this.currentInput.slice(0, -1);
                    tree.focusNode.parent ? this.center(tree.focusNode.parent, "left") : 0;
                }
                d3.event.preventDefault();
                break;
            }
            case 9: // TAB
            case 39: { // Right Arrow
                this.currentInput = "";
                if (tree.focusNode.children) {
                    this.center(tree.focusNode.children[this.currentFocusChildIndex], "right");
                }
                else {
                    // jump from EOS to SOS
                    this.mouseover(this.root, this.getNodeSelection(this.root));
                    this.center(this.root, "");
                }
                d3.event.preventDefault();
                this.currentFocusChildIndex = 0;
                if (this.focusNode.data.name === Constants.EOS && !this.focusNode.data.isCandidate) {
                    this.click(this.focusNode, this.getNodeSelection(this.focusNode))
                }
                d3.event.view.event.preventDefault();
                break;
            }
            case 38: { // Up Arrow
                if (this.focusNode.children) {
                    this.currentFocusChildIndex = (this.currentFocusChildIndex - 1) % this.focusNode.children.length;
                    if (this.currentFocusChildIndex < 0) {
                        this.currentFocusChildIndex = this.focusNode.children.length - 1;
                    }
                }
                break;
            }
            case 40: { // Down Arrow
                if (this.focusNode.children) {
                    this.currentFocusChildIndex = (this.currentFocusChildIndex + 1) % this.focusNode.children.length;
                }
                break;
            }
        }

        var key = d3.event.key;
        if (this.focusNode.data.name !== Constants.EOS && key.length === 1 && !(key == " " && this.currentInput.length === 0)) {
            this.currentInput += key;

            var node = this.getBeamNode(this.treeData, this.getPathList(this.focusNode).slice(1));
            if (!this.focusNode.data.isEdit && !this.hasEditChild(this.focusNode)) {
                this.currentInput = key;
                node.children.push({
                    attn: [],
                    name: this.currentInput,
                    children: [],
                    logprob: -3,
                    candidates: [],
                    isCandidate: true,
                    isEdit: true
                });
            } else if (node && this.focusNode.data.isEdit) { // TODO: this.focusNode is wrong after manual text edit in translation-box
                node.name = this.currentInput;
            }
            this.updateData(this.treeData);
            if (!this.focusNode.data.isEdit) {
                this.center(this.getEditChild(this.focusNode), "right");
            }
        } else if (this.currentInput.length > 0) {
            var node = this.getBeamNode(this.treeData, this.getPathList(this.focusNode).slice(1));
            node.name = this.currentInput;
            this.updateData(this.treeData);
        } else if (this.hasEditChild(this.focusNode)) {
            var node = this.getBeamNode(this.treeData, this.getPathList(this.focusNode).slice(1));
            this.removeEditChild(node);
            this.updateData(this.treeData);
            this.center(this.focusNode, "")
        }
        this.updateData(this.treeData);

        d3.select('#currentBeamInput').text("Simply type for correction " + this.currentInput);

        this.that.updateAttentionViewWeights(this.that.attention.slice(), d3);
    }

    // Creates a curved (diagonal) path from parent to the child node
    diagonal(s, d) {

        var path = `M ${s.y} ${s.x}
            C ${(s.y + d.y) / 2} ${s.x},
              ${(s.y + d.y) / 2} ${d.x},
              ${d.y} ${d.x}`

        return path
    }

    getNodeId(d) {
        var path = [];

        while (d) {
            if (d.data.isEdit) {
                path.push("<EDIT>")
            } else {
                path.push(d.data.name.replace(/"/g, "&quot;").replace(/'/g, "&apos;"));
            }
            d = d.parent;
        }

        return path.reverse().join("#");
    }

    getPathList(d) {
        var path = [];

        while (d) {
            if (d.data.isEdit) {
                path.push("<EDIT>")
            } else {
                path.push(d.data.name);
            }
            d = d.parent;
        }

        return path.reverse();
    }

    getPath(d) {
        var path = [];

        while (d) {
            path.push(d.data.is_unk ? "UNK" : d.data.name);
            d = d.parent;
        }

        return path.reverse().slice(0, -1).join(" ");
    }

    getTranslation(d) {
        var path = [];

        while (d) {
            path.push(this.that.decodeText(d.data.name));
            d = d.parent;
        }

        return path.reverse().slice(1);
    }

    getRawTranslation(d) {
        var path = [];

        while (d) {
            path.push(d.data.name);
            d = d.parent;
        }

        return path.reverse().slice(1);
    }

    getRoot(d) {
        var root = d;

        while (root.parent) {
            root = root.parent;
        }

        return root;
    }

    getBeamAttention(d) {
        var attention = [];

        while (d && d.data.name !== Constants.SOS) {
            attention.push(d.data.attn);
            d = d.parent;
        }
        return attention.reverse();
    }

    resetGoldenHypothesis(d) {
        if (!d.data) {
            return;
        }

        d.data.is_golden = false;

        if (!d.children || d.children.length == 0) {
            return;
        }

        for (var i = 0; i < d.children.length; i++) {
            this.resetGoldenHypothesis(d.children[i]);
        }
    }

    setGoldenHypothesis(d) {
        this.resetGoldenHypothesis(this.getRoot(d));
        var node = d;
        while (node) {
            node.data.is_golden = true;
            node = node.parent;
        }
        this.update(this.root);
    }

    lastNodeOfGoldenHypothesis(d) {
        var node = d;
        if (!node.data.is_golden) {
            return d;
        }
        while (node.children && node.children.length > 0) {
            var goldenChildFound = false
            var childId = -1;
            while (!goldenChildFound) {
                childId++;
                goldenChildFound = node.children[childId].data.is_golden;
            }
            node = node.children[childId];
        }
        return node;
    }

    lastNodeOfBestHypothesis(d) {
        var node = d;
        while (node.children && node.children.length > 0) {
            node = node.children[0];
        }
        return node;
    }

    resetGoldenHypothesisBeam(beam) {
        if (!beam) {
            return;
        }
        beam.is_golden = false

        for (var i = 0; i < beam.children.length; i++) {
            this.resetGoldenHypothesisBeam(beam.children[i]);
        }
    }

    setGoldenHypothesisBeam(beam, hypothesis) {
        if (hypothesis.length === 0) {
            return;
        }

        beam.is_golden = true
        for (var i = 0; i < beam.children.length; i++) {
            if (beam.children[i].name === hypothesis[0]) {
                this.setGoldenHypothesis(beam.children[i])
            }
        }
    }

    countNumHypothesis() {
        return this.numHypothesis(this.root);
    }

    numHypothesis(d) {
        var node = d;
        if (!node) {
            return 1;
        }

        var count = 0;
        for (var child of node.children) {
            if (child.data.name === Constants.EOS) {
                return count + 1;
            }
            else {
              count += this.numHypothesis(child);
            }
        }
        return count;
    }

    buildColorLegend() {
        var w = 90;
        var h = 40;
        var legend = this.svg.append("defs")
            .append("svg:linearGradient")
            .attr("id", "gradient")
            .attr("x1", "0%")
            .attr("y1", "100%")
            .attr("x2", "100%")
            .attr("y2", "100%")
            .attr("spreadMethod", "pad");

        var data = [];
        for (var i = 0.0; i < this.colors.length; i++) {
            data.push({offset: (i / this.colors.length) * 100 + "%", color: this.colors[i]});
        }
        legend.selectAll("stop")
            .data(data)
            .enter().append("stop")
            .attr("offset", function (d) {
                return d.offset;
            })
            .attr("stop-color", function (d) {
                return d.color;
            });

        var rect = d3.select('#tree-vis').append("g")
            .attr("transform", "translate(10, 20)");

        this.colorLegend = rect;

        rect
            .append("rect")
            .attr("width", w)
            .attr("height", h - 30)
            .style("fill", "url(#gradient)")
            .attr("transform", "translate(0,10)");

        rect
            .append("text")
            .text("Word Probability")
            .style("font-size", "12px");

        rect
            .append("foreignObject")
            .attr("x", 0)
            .attr("y", 465 - 90)
            .attr("width", "500px")
            .attr("height", "30px")
            .html("<span>Use <kbd>Ctrl</kbd> + <kbd>ENTER</kbd> to accept currently selected Best Translation</span>")
            .style("font-size", "12px");

        rect
            .append("foreignObject")
            .attr("x", 0)
            .attr("y", 490 - 90)
            .attr("width", "300px")
            .attr("height", "30px")
            .html("<span>Type any <kbd>key</kbd> for custom correction</span>")
            .style("font-size", "12px");

        rect
            .append("foreignObject")
            .attr("x", 0)
            .attr("y", 515 - 90)
            .attr("height", "30px")
            .attr("width", "400px")
            .html("<span>Select/Edit with <kbd>ENTER</kbd> or click, delete with <kbd>BACKSPACE</kbd></span>")
            .style("font-size", "12px");

        rect
            .append("foreignObject")
            .attr("x", 0)
            .attr("y", 540 - 90)
            .attr("height", "30px")
            .attr("width", "300px")
            .html("<span>Navigate with <kbd>TAB</kbd> and <kbd>←</kbd> <kbd>→</kbd> <kbd>↑</kbd> <kbd>↓</kbd></span>")
            .style("font-size", "12px");

        rect
            .append("foreignObject")
            .attr("x", 500)
            .attr("y", 550 - 90)
            .attr("height", "30px")
            .attr("width", "300px")
            .html('')
            .style("font-size", "12px");

        rect
            .append("text")
            .attr("x", 120)
            .attr("y", 0)
            .text("Best Translation")
            .style("font-size", "12px");

        rect.append("circle")
            .attr("cx", 160)
            .attr("cy", 15)
            .attr("r", 8)
            .style("fill", "none")
            .style("stroke", "#f6f6f6")
            .style("stroke-width", "3px");

        rect.append("line")
            .attr("x1", 130)
            .attr("x1", 130)
            .attr("y1", 15)
            .attr("x2", 150)
            .attr("y2", 15)
            .style("stroke", "#fed330")
            .style("stroke-width", "3px");

        rect.append("line")
            .attr("x1", 170)
            .attr("y1", 15)
            .attr("x2", 190)
            .attr("y2", 15)
            .style("stroke", "#fed330")
            .style("stroke-width", "3px");
    }

    getTransformation(transform) {
        // Create a dummy g for calculation purposes only. This will never
        // be appended to the DOM and will be discarded once this function
        // returns.
        var g = document.createElementNS("http://www.w3.org/2000/svg", "g");

        // Set the transform attribute to the provided string value.
        g.setAttributeNS(null, "transform", transform);

        // consolidate the SVGTransformList containing all transformations
        // to a single SVGTransform of type SVG_TRANSFORM_MATRIX and get
        // its SVGMatrix.
        var matrix = g.transform.baseVal.consolidate().matrix;

        // Below calculations are taken and adapted from the private function
        // transform/decompose.js of D3's module d3-interpolate.
        var {a, b, c, d, e, f} = matrix;   // ES6, if this doesn't work, use below assignment
        var scaleX, scaleY, skewX;
        if (scaleX = Math.sqrt(a * a + b * b)) a /= scaleX, b /= scaleX;
        if (skewX = a * c + b * d) c -= a * skewX, d -= b * skewX;
        if (scaleY = Math.sqrt(c * c + d * d)) c /= scaleY, d /= scaleY, skewX /= scaleY;
        if (a * d < b * c) a = -a, b = -b, skewX = -skewX, scaleX = -scaleX;
        return {
            translateX: e,
            translateY: f,
            rotate: Math.atan2(b, a) * 180 / Math.PI,
            skewX: Math.atan(skewX) * 180 / Math.PI,
            scaleX: scaleX,
            scaleY: scaleY
        };
    }
}
