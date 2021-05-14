import * as d3 from 'd3';
import {Component} from '@angular/core';
import {Constants} from '../../constants';

export class MetricInfo {

    svg;
    sentence;

    constructor(private that: any) {
    }

    build() {
        this.update();
    }

    update() {
        var that = this.that;

        this.sentence = that.sentence;

        var width = 29;
        var height = 40;

        // append the svg object to the body of the page
        // appends a 'group' element to 'svg'
        // moves the 'group' element to the top left margin
        d3.select("#barchart-vis-" + this.sentence.id).selectAll('*').remove();
        this.svg = d3.select("#barchart-vis-" + this.sentence.id)
            .attr("width", width)
            .attr("height", height)
            .attr("id", "barchart-vis-" + this.sentence.id)
            .append("g");

        var background = this.svg.append("rect")
                                          .attr("x", 0)
                                          .attr("y", 0)
                                          .attr("fill", "none")
                                          .attr("width", 29)
                                          .attr("height", height);

        var rectangleConfidence = this.svg.append("rect")
                                          .attr("x", 0)
                                          .attr("y", height - height * this.sentence.score['confidence'])
                                          .attr("fill", "teal")
                                          .attr("width", 5)
                                          .attr("height", height * this.sentence.score['confidence']);
        rectangleConfidence.append("svg:title").text("Confidence: " + this.sentence.score['confidence']);

        var rectangleCoveragePenalty = this.svg.append("rect")
                                          .attr("x", 8)
                                          .attr("y", height - height * this.sentence.score['coverage_penalty'] / 100)
                                          .attr("fill", "teal")
                                          .attr("width", 5)
                                          .attr("height", height * this.sentence.score['coverage_penalty'] / 100);
        rectangleCoveragePenalty.append("svg:title").text("Coverage Penalty: " + this.sentence.score['coverage_penalty'] );

        var rectangleLength = this.svg.append("rect")
                                          .attr("x", 16)
                                          .attr("y", height - height * this.sentence.score['length'] / 50)
                                          .attr("fill", "teal")
                                          .attr("width", 5)
                                          .attr("height", height * this.sentence.score['length'] / 50);
        rectangleLength.append("svg:title").text("Length: " + this.sentence.score['length']);

        var rectangleKeyphrases = this.svg.append("rect")
                                          .attr("x", 24)
                                          .attr("y", height - height * this.sentence.score['keyphrase_score'] / 10)
                                          .attr("fill", "teal")
                                          .attr("width", 5)
                                          .attr("height", height * this.sentence.score['keyphrase_score'] / 10);
        rectangleKeyphrases.append("svg:title").text("Keyphrases: " + this.sentence.score['keyphrase_score']);
    }
}
