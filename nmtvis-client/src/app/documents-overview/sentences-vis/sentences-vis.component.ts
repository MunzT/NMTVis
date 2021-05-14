import {Component, OnInit, OnChanges, AfterViewInit, SimpleChange, SimpleChanges, Input, Output, EventEmitter} from '@angular/core';
import * as d3 from 'd3';

@Component({
    selector: 'app-sentences-vis',
    templateUrl: './sentences-vis.component.html',
    styleUrls: ['./sentences-vis.component.css']
})
export class SentencesVisComponent implements OnInit, OnChanges, AfterViewInit {

    @Input()
    private selectedSentence;
    @Output()
    private selectedSentenceChange = new EventEmitter<string>();
    @Input()
    sentences = [];
    @Input()
    metric = "";
    @Input()
    color = "orange";
    @Input()
    shortname = "";
    @Output()
    onSort = new EventEmitter<any>();

    sortAscending = true;


    constructor() {
    }

    ngOnInit() {

    }

    ngAfterViewInit() {
        this.updateSentences(this.sentences);
    }

    ngOnChanges(changes: SimpleChanges) {
        const sentences: SimpleChange = changes.sentences;

        if (changes.sentences) {
            this.updateSentences(sentences.currentValue);
        }
        if (changes.selectedSentence) {
            if (!changes.selectedSentence.currentValue) {
                return;
            }
            var i = this.sentences.indexOf(changes.selectedSentence.currentValue);
            d3.selectAll("#sentences-vis-" + this.metric + " .background-sentence-bar").classed("selected-sentence-bar", false);
            d3.select("#" + this.metric + "-background-sentence-bar-" + i).classed("selected-sentence-bar", true);
        }
    }

    onSortClick() {
        this.onSort.emit([this.metric, this.sortAscending]);
        this.sentences.sort((a, b) => {
            if (this.sortAscending) {
                return a["score"][this.metric] - b["score"][this.metric];
            } else {
                return b["score"][this.metric] - a["score"][this.metric];
            }
        });
        this.sortAscending = !this.sortAscending;
    }

    updateSentences(sentences) {
        if (!sentences || sentences.length == 0) {
            return;
        }

        var that = this;
        that.selectedSentence = sentences[0];
        var margin = {top: 15, right: 50, bottom: 10, left: 0},
            width = 560 - margin.left - margin.right,
            height = 40 - margin.top - margin.bottom;

        // append the svg object to the body of the page
        // appends a 'group' element to 'svg'
        // moves the 'group' element to the top left margin
        d3.selectAll("#sentences-vis-" + this.metric).remove();
        var svg = d3.select("#sentences-vis-box-" + this.metric).append("svg")
            .attr("width", width + margin.right + margin.left)
            .attr("height", height + margin.top + margin.bottom)
            .attr("id", "sentences-vis-" + this.metric)
            .append("g")
            .attr("transform", "translate("
                + margin.left + "," + margin.top + ")");

        var zoom = d3.zoom().on("zoom", function () {
            var deltaY = d3.event.sourceEvent.deltaY;

            var i = sentences.indexOf(that.selectedSentence);
            var direction = deltaY < 0 ? 1 : -1;
            i = ((i + direction) % sentences.length + sentences.length) % sentences.length;
            that.selectedSentence = that.sentences[i];
            that.selectedSentenceChange.emit(that.selectedSentence);
            that.scrollParentToChild(document.getElementById("document-scroll"),
                document.getElementById("sentence-" + that.selectedSentence.id));
        });
        svg.call(zoom);

        var title = svg.append("text").text(this.shortname).attr("x", width + 2).attr("y", 15).style("text-anchor", "start")
            .on("click", function () {

            });

        var data = [];
        for (var i = 0; i < sentences.length; i++) {
            data.push(sentences[i]["score"][this.metric]);
        }

        // set the ranges
        var x = d3.scaleBand()
            .range([0, width])
            .padding(0.2);
        var y = d3.scaleLinear()
            .range([height - 5, 0]);

        // Scale the range of the data in the domains
        x.domain(data.map(function (d: any, i: any) {
            return i;
        }));
        y.domain([d3.min(data, function (d) {
            return d;
        }), d3.max(data, function (d) {
            return d;
        })]);

        // append the rectangles for the bar chart
        var barEnter = svg.selectAll(".bar")
            .data(data)
            .enter();

        barEnter.append("rect")
            .attr("x", function (d: any, i: any) {
                return x(i);
            })
            .attr("class", "background-sentence-bar")
            .attr("width", x.bandwidth())
            .attr("y", function (d) {
                return 0;
            })
            .attr("height", function (d) {
                return height;
            })
            .attr("id", function (d, i) {
                return that.metric + "-background-sentence-bar-" + i;
            })
            .on("mouseover", function (d, i) {
                that.selectedSentence = that.sentences[i];
                that.selectedSentenceChange.emit(that.selectedSentence);
                d3.select(this).classed("selected-sentence-bar", true);

                that.scrollParentToChild(document.getElementById("document-scroll"),
                    document.getElementById("sentence-" + that.selectedSentence.id));
            })
            .on("mouseout", function () {
                d3.select(this).classed("selected-sentence-bar", false);
            });

        barEnter.append("rect")
            .attr("class", "sentence-bar")
            .attr("x", function (d: any, i: any) {
                return x(i);
            })
            .attr("width", x.bandwidth())
            .attr("y", function (d) {
                return y(d);
            })
            .attr("height", function (d) {
                return height - y(d);
            })
            .style("fill", this.color)
            .style("pointer-events", "none");
    }

    scrollParentToChild(parent, child) {

        // Where is the parent on page
        var parentRect = parent.getBoundingClientRect();
        // What can you see?
        var parentViewableArea = {
            height: parent.clientHeight,
            width: parent.clientWidth
        };

        // Where is the child
        var childRect = child.getBoundingClientRect();
        // Is the child viewable?
        var isViewable = (childRect.top >= parentRect.top) && (childRect.top <= parentRect.top + parentViewableArea.height);

        // if you can't see the child try to scroll parent
        if (!isViewable) {
            // scroll by offset relative to parent
            parent.scrollTop = (childRect.top + parent.scrollTop) - parentRect.top
        }
    }

    ngAfterContentInit() {
    }

}
